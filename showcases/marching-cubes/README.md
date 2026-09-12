# NVIDIA marching cubes — implicit mesh

[Open in the sandbox](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=marching).

This showcase compiles the original NVIDIA `classifyVoxel`, `compactVoxels` and
shared `generateTriangles` kernels into WGSL, then renders their position and
normal buffers directly with Three.js. Drag the mesh to orbit. The shader menu
contains each NVIDIA pass and the two project scan shaders.

The source comes from CUDA Samples commit
`5443602d89ed99aede2e4b7bf329daddeadb320e`,
`cpp/5_Domain_Specific/marchingCubes/marchingCubes_kernel.cu`, its `defines.h`
and the original `Common/helper_math.h` interpolation helpers. Device function
bodies and copyright notices are retained; desktop host code is omitted.
`tables.json` contains the original `triTable` and `numVertsTable` from
`tables.h` (`X` is 255), under the same NVIDIA BSD-3-Clause license.
See the full notice in `kernel.cu` and the repository's third-party notices.

`pipeline.json` selects the existing upstream implicit-field branch through
compile definitions: `SAMPLE_VOLUME=0`, `USE_SHARED=1`, `NTHREADS=32` and
`SKIP_EMPTY_VOXELS=1`. `_DEFINES_H_=1` skips the embedded header's defaults.
These definitions override the defaults visible in the source editor; CUDA
function bodies are unchanged. The desktop program and sampled-volume
`generateTriangles2` branch are not supported by this showcase yet.

The default 16³ field at isovalue 0.5 produces 1,024 active voxels, 6,240
vertices and 2,080 triangles. The sequence is:

1. Classify voxels with the original CUDA kernel.
2. Compute two exclusive unsigned scans on the GPU.
3. Compact active voxel indices with the original CUDA kernel.
4. Generate positions and gradients with the original shared-memory kernel.
5. Render those GPU buffers as a lit triangle mesh.

The scans replace the desktop host-library primitive. They are MIT project
CUDA kernels, compiled by the same frontend; they are not NVIDIA sample code.
Their WGSL entries are labelled **runtime scan CUDA helper** in the comparison
view, where the left editor continues to contain the NVIDIA source.

Only two uint totals (8 bytes) are read while executing the pipeline. The
sandbox subsequently inspects final positions to frame the camera. It does
not compute triangles in JavaScript or upload positions/normals after CUDA
writes them. Orbiting performs no compute dispatch or simulation-data transfer.

To change the isovalue, edit `isoValue` in both the classification and triangle
steps under Launch settings, then compile and run. An empty surface is valid.
Dimensions, capacities, masks, shifts and launch dimensions must agree; the
sandbox validates resource limits and types but cannot prove arbitrary
user-authored kernel indexing is correct.

The generic pipeline format declares buffers, textures, dispatches, scans and
the preview in JSON. A scalar or group dimension can read a uint control buffer
with `{ "buffer": "occupiedTotal", "divideCeil": 32 }`. Limits are 24 steps,
32 buffers, 64 MiB of buffer storage and 16 MiB of logical texture data. The
current mesh preview requires float4 position and normal buffers and a vertex
count divisible by three. Computation runs on demand; the Animate checkbox is
disabled for this pipeline.

Hardware validation compares all 49,920 position/gradient components with
native CUDA captures. Maximum absolute errors are approximately 2.98e-8 for
positions and 9.31e-10 for gradients, below the 1e-6 tolerance. This is numerical
agreement, not bit-exact output or a performance measurement. The native
captures are expected results only, never pipeline inputs.

Run `npm run build` then `node scripts/test-marching-sandbox.mjs` with NVIDIA
hardware and Edge. Results and a screenshot are in
`reports/marching-cubes-sandbox-check.json` and
`reports/marching-cubes-sandbox.png`. The test covers native agreement, generated
passes, unchanged source, camera motion without compute/data transfers, an empty
surface and mobile layout.
