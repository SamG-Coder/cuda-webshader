# NVIDIA FFT convolution

[Open in the sandbox](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=fft-convolution).

Runs the original convolutionFFT2D test0 path: pad the filter and input, transform both with R2C, multiply and normalize spectra, then invert with C2R. The input is the original 2000 × 2000 random field; the filter is 7 × 6 centred at (4,3); both FFTs are 2048 × 2048. Inputs capture Windows srand(2010), rand()%16 in the same order as NVIDIA main.cpp. Random sequences are C-library specific, so binary captures ensure native and browser use identical inputs.

The preview displays the complete 2048-square output, including the last 48 padding rows and columns. Raw convolution values are mapped to grayscale over [0,5055], the nonnegative input bound 15 multiplied by the filter weight sum. Display scaling does not change CUDA source or GPU results. The original sample has random numerical input rather than a scene or photograph.

The original padKernel_kernel, padDataClampToBorder_kernel, modulateAndNormalize_kernel and mulAndScale bodies are retained. The original USE_TEXTURE and LOAD_FLOAT macro select element reads from float linear textures. The host pipeline and real FFT adapter are MIT project code; cuFFT itself does not run in the browser. The comparison menu exposes the layout and FFT adapter shaders as well as the original kernels.

Native and independent CPU validation cover 37 × 19 input with a 64 × 32 FFT and the original full size. All 4,194,304 full-size output values are finite and compared with native CUDA: relative L2 1.99794e-7, maximum absolute error 0.0029296875. Original CPU error accounting yields relative L2 1.38515e-7, below NVIDIA's 1e-6 criterion, across all four million input-domain pixels including clamped borders. No intermediate CPU readback occurs. The sandbox test checks output, shader stages, source preservation, grayscale contrast, mobile layout and switching examples on NVIDIA Blackwell hardware.

Reproduce native capture with tests/fft-convolution-native.cu, the pinned original source directory on the include path and cuFFT linkage. Run node scripts/test-fft-convolution.mjs and node scripts/test-fft-convolution-sandbox.mjs after npm run build. Reports are in reports/fft-convolution-check.json, fft-convolution-native.txt and fft-convolution-sandbox-check.json.

NVIDIA source: cuda-samples revision 5443602d89ed99aede2e4b7bf329daddeadb320e, cpp/5_Domain_Specific/convolutionFFT2D/convolutionFFT2D.cuh. The device source retains its BSD-3-Clause notice. The sample's two custom complex-transform paths (test1/test2) are separate candidates and are not represented as complete by this preset.
