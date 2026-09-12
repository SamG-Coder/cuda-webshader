# NVIDIA optical flow

[Run the sandbox](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=optical).

The complete HSOpticalFlow sequence runs from six unchanged NVIDIA device kernels: AddKernel, ComputeDerivativesKernel, DownscaleKernel, JacobiIteration, UpscaleKernel and WarpingKernel. The original BSD-3-Clause notice and integer template declaration remain in kernel.cu.

The preset uses NVIDIA's frame10.ppm and frame11.ppm at 640 × 480. frame-0.bin and frame-1.bin contain little-endian float32 red-channel values divided by 255, matching the original LoadImageAsFP32 host loader. Input data and device functions come from cuda-samples revision 5443602d89ed99aede2e4b7bf329daddeadb320e, cpp/5_Domain_Specific/HSOpticalFlow.

The original configuration is retained: alpha 0.2, five pyramid levels, three warps per level and 500 Jacobi iterations per warp (7,500 solver dispatches). Downscaling, mirrored linear texture reads, warping, derivative calculation, repeated Jacobi steps, flow addition and upscaling run on WebGPU. The host uses separate output buffers for AddKernel to satisfy WebGPU's binding alias rules, then alternates those buffers. Kernel bodies are not changed.

The declarative pipeline has 184 top-level steps, with bounded repeats expanding to 7,669 GPU operations. Four update buffers are cleared before each warp; intermediate images and motion fields remain on the GPU. Six distinct shader variants are compiled and reused. Only final U and V are read back for the preview.

Colour is a display operation: hue represents direction (right red, down lime, left cyan, up violet); saturation represents displacement, reaching full colour at 8 pixels. White means zero motion. The display does not alter the solver output. Edit preview.scale in the launch JSON to change colour sensitivity.

Native reference: tests/optical-full-native.cu calls the original ComputeFlowCUDA host implementation with the same inputs and settings. All 614,400 final U/V components match the native captures exactly on the tested NVIDIA Blackwell adapter. Regression tolerances allow maximum error 0.00002 and RMS 0.000002 pixels per component. The original Jacobi stage is separately tested at iterations 1, 8 and 500 with two tile sizes and padded rows.

The sandbox test checks final native values, six shader stages, unchanged source, coloured output, mobile layout and switching presets. See reports/optical-flow-check.json and reports/optical-sandbox-check.json. No software WebGPU adapter or performance speedup claim is used.
