# NVIDIA surface write

[Open the surface-write sandbox](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=surface).

This runs both original kernels from NVIDIA CUDA Samples revision
`5443602d89ed99aede2e4b7bf329daddeadb320e`,
`cpp/0_Introduction/simpleSurfaceWrite/simpleSurfaceWrite.cu`.
`surfaceWriteKernel`, `transformKernel` and the original `teapot512.pgm` image
retain NVIDIA's BSD-3-Clause terms. Kernel bodies are unchanged.

The host decodes the input PGM into float pixels and uploads them once. The
first kernel writes those pixels through `surf2Dwrite` into an `r32float`
storage texture. The second kernel samples that same GPU texture and writes
the rotated image into the original input buffer, which is no longer needed.
Both dispatches are submitted together; no intermediate image is read back
or uploaded. The final float output is displayed as grayscale.

Launch settings use 64×64 blocks of 8×8 threads for the 512×512 image.
The second pass's `scalars.theta` controls rotation in radians, initially 0.5.
The shader selector exposes both generated WGSL programs. Change settings and
click Compile & Run to recompute.

The initial surface subset supports float `surf2Dwrite` with byte offset
`globalX * 4`, row `globalY`, and `cudaBoundaryModeTrap`. The compiler proves
this coordinate pattern through unmodified local initializers; mutated or
escaped coordinates and other boundary modes are rejected. The runtime checks
the entire dispatch fits the surface before submitting it. It does not silently
replace CUDA trapping with WebGPU's out-of-range write discard. Arbitrary surface
coordinates, surface reads and surface helper parameters remain unsupported.

Additional sandbox passes can now bind existing textures or surfaces with
matching formats. A writable surface cannot alias another resource binding in
the same dispatch, but can be sampled by a later pass. Float filtering requires
`float32-filterable`. The original transform kernel has no bounds guard; its
dispatch must exactly cover its output buffer, as the preset does.

Native CUDA and NVIDIA WebGPU both verify that every surface texel exactly
matches the input image. Three complete rotated outputs (0.5, 0 and −0.7 radians)
are compared with native CUDA, including sixteen output guards. Observed
maximum float differences are approximately 0.00164, 0 and 0.00164; tolerance
is 0.005 for texture interpolation precision. Runtime counters verify there is
no intermediate CPU image transfer. An oversized surface dispatch is rejected.

The built sandbox compares every displayed pixel at the default angle and after
editing the angle to zero. It also checks both-pass shader selection, mobile
layout and oversized-dispatch rejection. See `reports/surface-write-native.txt`,
the surface-write entry in `reports/nvidia-regression-gpu.json`, and
`reports/surface-write-sandbox-check.json`. These are correctness tests, not
CUDA-versus-WebGPU performance claims or execution of the desktop host APIs.
