# NVIDIA stereo disparity

[Run the sandbox](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=stereo).

The original NVIDIA stereoDisparityKernel compares 17 horizontal shifts (-16 through 0) over a 17 × 17 support window. Shared memory accumulates the matching costs. Its unchanged __usad4 device helper uses inline PTX to sum the four packed colour-byte differences.

kernel.cu retains both original device bodies and their BSD-3-Clause notice. Only the header guard, cooperative-groups include and CPU host reference are excluded from the extracted browser source. The compiler recognises the supported cooperative-groups operations and the exact SIMD SAD instruction. The host supplies launch configuration and r32uint textures with unnormalised nearest clamp sampling.

image-0.bin and image-1.bin contain the RGB pixels from NVIDIA's stereoDisparity/data/stereo.im0.640x533.ppm and stereo.im1.640x533.ppm, packed into little-endian uint32 words with a zero fourth byte. The input source is the pinned cuda-samples checkout used throughout this repository (5443602d89ed99aede2e4b7bf329daddeadb320e). Both native CUDA and WebGPU receive those same words.

The output stores d + 8 in unsigned words, so negative values wrap. The display interprets those words as signed 32-bit values and maps -8..8 to black..white. Darker pixels indicate larger horizontal shifts. This is a disparity image, not calibrated metric depth or a reconstructed 3D mesh. No user CUDA code is modified for display.

Native validation compares every output word exactly on the original 640 × 533 image pair (341,120 pixels) and a synthetic 67 × 19 shifted image (1,273 pixels). The latter exercises non-full edge blocks, borders, high bytes and a known horizontal offset. The separate SIMD test compares 4,100 native instruction results including accumulator wraparound. The sandbox test checks all output words and all displayed grayscale values, unchanged editor source and mobile layout, using NVIDIA hardware rather than a software adapter.

See reports/stereo-check.json and reports/stereo-sandbox-check.json at the repository root. No performance speedup is claimed.
