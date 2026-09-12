# FFT convolution candidate

Selected NVIDIA convolutionFFT2D after optical flow. The original default case uses a 2000 × 2000 real input, a 7 × 6 kernel centred at (4,3), and 2048 × 2048 FFTs. Target the original R2C/C2R convolution path, then inspect the two explicit complex-transform variants in main.cpp.

First compiler/runtime step: raw float, float2 and float4 linear textures. createLinearTexture accepts Float32Array with 1, 2 or 4 components per record. The texture is laid out across device-width rows while preserving logical record count. Scalar float fetches retain compatibility with normalized-byte textures; vector fetch types and helper aliases propagate through compilation. Linear float bindings use unfilterable-float texture loads, so no floating-point filtering feature is required for this operation.

Complete primary-call macros with fixed/free arguments now preserve the original LOAD_FLOAT(i) tex1Dfetch<float>(texFloat, i) expression. Existing forwarding macro chains retain their previous expansion rules. Non-primary textual expressions and unparenthesized parameter arithmetic remain rejected when they cannot preserve CUDA precedence.

Native validation checks 32,775 lookups each of float, float2 and float4 (229,425 scalar components) against CUDA. All are exact, including negative/out-of-range indices, vector alias helpers and multi-row/padded-tail storage. See linear-float-check.json and linear-float-native.txt.

The original padKernel_kernel, padDataClampToBorder_kernel and modulateAndNormalize_kernel bodies plus mulAndScale are retained in tests/fft-convolution-kernels.cuh and compile. Their complete convolution pipeline is not yet running. Remaining work includes 2048-point FFT axes, original-size memory budgets, native FFT convolution comparisons and a sandbox result preview. No new runnable showcase is claimed yet.

## 2048-point transform milestone

The runtime complex and real FFT APIs now support axes through 2048 elements. One workgroup still handles each row or column, with at most 1024 threads and 16 KiB of shared float2 storage. Each active lane computes a complete butterfly pair; larger axes distribute loads and stores across lanes without assuming one thread per element.

Native cuFFT captures cover 2048 × 8, 8 × 2048 and 2048 × 2048. At the square size, relative spectrum L2 error is 5.3833e-7 and maximum normalized native-inverse error is 1.37091e-6. A GPU forward/inverse round trip has maximum input-domain error 2.38419e-6 for input values in [-1,1]. The large-transform roundtrip check allows 4e-6; smaller transforms retain their existing 2e-6 limit. Native forward relative error (2e-6), native inverse error (2e-6), row padding and guard checks remain in force. See large-fft-check.json.

This is a runtime milestone. The sandbox's FFT-size and memory limits still need updating for the original 2000-square convolution preset, followed by the full native convolution comparison and showcase preview.

## Complete default test0 pipeline

The original 2000-square path now passes end to end on NVIDIA Blackwell WebGPU, against both original native CUDA and independent spatial convolution at all four million input pixels. See fft-convolution-check.json and fft-convolution-sandbox-check.json. The sandbox preset retains original kernels and exposes all generated passes. It uses bounded raw float textures, 2048-point FFT axes, at most 4,194,304 records per buffer and a 128 MiB declared-buffer budget. Texture storage retains its 64 MiB budget; FFT scratch storage is additional runtime memory. The grayscale range [0,5055] affects display only. The original-size numerical showcase has its own main-page card linking directly to the sandbox.

The two custom C2C transform variants in original main.cpp remain next to inspect and verify; this preset implements test0 only.

## Custom-transform compiler stages

The three original spPostprocess2D_kernel, spPreprocess2D_kernel and spProcess2D_kernel bodies now compile and execute unchanged. Compiler support adds __ffs with CUDA's one-based bit position/zero semantics, sincosf and __sincosf output addresses to mutable local float values/components (including reference components), and call-site type aliases inside expression macros. Phase is captured once before either destination is written. Same-output aliasing, const destinations, non-float destinations and unsupported pointer storage are rejected.

Generated sincos uses split 2-pi argument reduction for |phase| <= 8192 and direct backend trigonometry outside that interval. This is floating-point translation rather than bit-identical emulation of CUDA's fast intrinsic. Over 8,195 phases in [-51.2,51.225], mathematical maximum error is 4.23846e-7 and standard CUDA sincosf error is 4.17233e-7. Fast __sincosf differs by up to 7.74860e-6 in this range, so the probe checks standard/math error at 2e-6 and fast-native difference at 1e-5 separately. __ffs matches all native values exactly, including every individual bit position, zero and signed combinations. See sincos-check.json.

All three original custom stages at DX=64, DY=32, padding=16 match native exactly for both directions, with padding and guard values preserved (24,584 computed scalar components across six captures). See fft-custom-check.json. These stage checks do not yet prove the full custom 2000-square convolution variants. Remaining work is GPU buffer-to-linear-texture copies, complex FFT pipeline declarations and explicit byte reinterpretation copies, followed by full native captures, original-size comparison and individual sandbox presets/cards. The already published default test0 showcase remains the complete path so far.
