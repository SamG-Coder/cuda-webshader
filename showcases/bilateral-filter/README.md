# NVIDIA bilateral filter

[Open the bilateral filter sandbox](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=bilateral).

The original `d_bilateral_filter` and its device helpers are extracted without body changes from NVIDIA CUDA Samples revision `5443602d89ed99aede2e4b7bf329daddeadb320e`, `cpp/5_Domain_Specific/bilateralFilter/bilateral_kernel.cu`. `nature.bmp` is copied byte for byte from `data/nature_monte.bmp`. The NVIDIA code and image retain the BSD-3-Clause license; original integration and tests are MIT.

The preset matches the upstream default: one pass, radius 5, spatial Gaussian delta 4 and colour-distance delta 0.1. It computes spatial weights on the host using the upstream formula, binds them as constant-array uniforms, and runs the original filter entirely on the GPU. This is the default single-pass image algorithm; the desktop OpenGL UI and optional multi-iteration ping-pong path are not implemented by this preset.

Edit `e_d` in Launch settings: a larger positive value blends across greater colour differences; smaller positive values preserve stronger colour separation. Radius `r` must be an integer from 0 through 31 and requires `2*r+1` corresponding Gaussian coefficients. For spatial delta `d`, coefficient `i` is `exp(-((i-r)^2)/(2*d*d))`. Keep `d` and `e_d` positive. The original early return supports partial blocks, tested on odd image dimensions.

The BMP decoder preserves the original sample loader’s BGR-to-RGB conversion, file row order and zero alpha. The GPU reads the same normalized input values as native CUDA. Native uses a normalized uchar4 texture; WebGPU uploads the decoded normalized values to an RGBA32 float texture. The image preview flips bottom-up BMP rows and makes alpha opaque only when presenting the final output. No image filtering runs in JavaScript outside the independent test reference.

Three cases compare every output channel and all guard elements against native CUDA and an independent bilateral-filter reference: 17 × 9 with radius 0, 31 × 19 with radius 2, and the original 640 × 480 image with radius 5. Maximum observed native/WebGPU error is one level out of 255; the test tolerance is two levels to accommodate transcendental and floating-point differences. The static sandbox also checks every displayed native pixel, the BMP orientation, an edited colour-distance parameter, shader comparison and mobile layout.

Evidence: `reports/bilateral-filter-native.txt`, the bilateral entry in `reports/nvidia-regression-gpu.json`, and `reports/bilateral-filter-sandbox-check.json`. These validate correctness, not native-versus-WebGPU performance. The native harness uses the original upstream BMP loader and original device functions.
