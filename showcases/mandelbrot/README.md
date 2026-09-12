# NVIDIA Mandelbrot and Julia
[Open Mandelbrot](../../sandbox.html?example=mandelbrot) · [Open Julia](../../sandbox.html?example=julia) · [Open accumulated frames](../../sandbox.html?example=mandelbrot-accumulated)

The sandbox compiles and runs the original NVIDIA `Mandelbrot0<float>` kernel. Its CUDA function bodies are unchanged. The source combines device declarations extracted from `Mandelbrot_kernel.cuh` and `Mandelbrot_cuda.cu`, preserving conditional branches and the original copyright notice, from CUDA Samples revision `5443602d89ed99aede2e4b7bf329daddeadb320e`.

## Supported scope
Mandelbrot and Julia float rendering, packed RGBA output, and sequential frame accumulation are verified. The source also contains helpers for other paths; their presence does not imply support. The secondary anti-alias kernel, double-single path, double precision and original desktop/OpenGL application are not exposed or validated here.

Launch settings are editable in the sandbox. The defaults use a 256 × 192 image, 16 × 16 threads per workgroup, 2 worker workgroups and 192 logical image tiles. The original kernel distributes those tiles between workers. If changing dimensions, also update buffer records, preview dimensions, gridWidth = ceil(width / 16) and numBlocks = gridWidth × ceil(height / 16).

The packed colors parameter uses bytes x/y/z/w in low-to-high order: RGB multipliers (3,5,7) are 0x00070503. The kernel writes zero alpha on its first frame; the generic preview displays the RGB image with opaque alpha and flips its rows for presentation. The accumulated preset runs a second dispatch on the same buffer with frame=1. It is two sequential frames, not continuous animation.

## Validation and floating-point behaviour
Six frames cover 33 × 25 Mandelbrot, 256 × 192 Mandelbrot and 256 × 192 Julia, each with an initial and accumulated frame. Every pixel agrees between hardware WebGPU, an independent float32 reference and native CUDA built with:

`nvcc -O3 --fmad=false -std=c++17 -arch=native tests/mandelbrot-native.cu -o .local/nvidia-checks/mandelbrot.exe`

This arithmetic setting matters. Default native multiply-add fusion changes escape iterations near fractal boundaries. The four larger fused frames differ at 79, 108, 44 and 93 pixels respectively. Both sets of captures are retained; exact agreement is claimed only for the matched unfused profile on the tested hardware. These are correctness checks, not performance measurements.

Native captures and fusion comparison are in `reports/mandelbrot-native-*.bin` and `reports/mandelbrot-floating-point.json`. GPU comparison is implemented in `tests/mandelbrot-gpu.js`; sandbox RGB presentation, edited colour/viewport, generated WGSL and mobile layout checks are in `scripts/test-mandelbrot-sandbox.mjs`.

## Licensing
NVIDIA CUDA source retains its BSD-3-Clause notice and original attribution comments, including DSFUN90 attribution. Original compiler, host integration, tests and vector illustration are MIT. See the root third-party notices.
