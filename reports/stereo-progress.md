# Stereo disparity candidate

Selected NVIDIA stereoDisparity after completing fluidsGL. The original device source is extracted into tests/stereo-kernels.cuh; only the header guard, include and host CPU reference are excluded. Device function bodies remain unchanged.

First blocker resolved: the original __usad4 helper uses inline PTX `vabsdiff4.u32.u32.u32.add`. The compiler now accepts this exact instruction with four positional registers, one named integer output, and three integer input constraints. It computes four unsigned byte differences, sums them with the 32-bit accumulator, and preserves wraparound. Inputs are evaluated once. Other opcodes, modifiers, constraint forms and clobbers fail explicitly; this is not general PTX support.

Semantics reference: [NVIDIA PTX ISA, SIMD video instructions](https://docs.nvidia.com/cuda/pdf/ptx_isa_9.0.pdf).

Native validation: tests/ptx-sad-native.cu executes the unchanged original helper for 4,100 inputs including full-range lanes and overflowing accumulators. Hardware WebGPU output matches every native value exactly and also matches an independent byte-sum oracle. See ptx-sad-check.json, ptx-sad-native.txt and captures. Full regression: 551 unit tests, 170 real NVIDIA GPU checks, no software adapter.

Next blocker: `tex2D<unsigned int>` on packed r32uint 2D images. The current compiler supports byte and float 2D sampling, but not this unsigned 32-bit format. The whole stereo kernel is not running yet; no showcase card is claimed until its complete output has been compared with native CUDA and its sandbox preview works.

## Complete kernel and sandbox

Unsigned 2D texture support now maps the original tex2D<unsigned int> calls to r32uint nearest/clamp pixel reads. Both original device bodies remain unchanged. The complete kernel matches native CUDA exactly for 341,120 original image pixels and 1,273 odd-dimension synthetic pixels. The standalone sandbox preview interprets the d+8 output as signed words and maps -8..8 to grayscale. All displayed pixel values are checked against the native output. Source, host configuration and limitations are documented in showcases/stereo/README.md.
