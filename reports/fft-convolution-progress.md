# FFT convolution candidate

Selected NVIDIA convolutionFFT2D after optical flow. The original default case uses a 2000 × 2000 real input, a 7 × 6 kernel centred at (4,3), and 2048 × 2048 FFTs. Target the original R2C/C2R convolution path, then inspect the two explicit complex-transform variants in main.cpp.

First compiler/runtime step: raw float, float2 and float4 linear textures. createLinearTexture accepts Float32Array with 1, 2 or 4 components per record. The texture is laid out across device-width rows while preserving logical record count. Scalar float fetches retain compatibility with normalized-byte textures; vector fetch types and helper aliases propagate through compilation. Linear float bindings use unfilterable-float texture loads, so no floating-point filtering feature is required for this operation.

Complete primary-call macros with fixed/free arguments now preserve the original LOAD_FLOAT(i) tex1Dfetch<float>(texFloat, i) expression. Existing forwarding macro chains retain their previous expansion rules. Non-primary textual expressions and unparenthesized parameter arithmetic remain rejected when they cannot preserve CUDA precedence.

Native validation checks 32,775 lookups each of float, float2 and float4 (229,425 scalar components) against CUDA. All are exact, including negative/out-of-range indices, vector alias helpers and multi-row/padded-tail storage. See linear-float-check.json and linear-float-native.txt.

The original padKernel_kernel, padDataClampToBorder_kernel and modulateAndNormalize_kernel bodies plus mulAndScale are retained in tests/fft-convolution-kernels.cuh and compile. Their complete convolution pipeline is not yet running. Remaining work includes 2048-point FFT axes, original-size memory budgets, native FFT convolution comparisons and a sandbox result preview. No new runnable showcase is claimed yet.
