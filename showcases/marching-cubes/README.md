# NVIDIA marching cubes — implicit and sampled-volume meshes

[Open the implicit field in the sandbox](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=marching).

[Open the Bucky volume in the sandbox](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=marching-volume).

This showcase compiles the original NVIDIA `classifyVoxel`, `compactVoxels` and
shared `generateTriangles` kernels into WGSL, then renders their position and
normal buffers directly with Three.js. Drag the mesh to orbit. The shader menu
contains each NVIDIA pass and the two project scan shaders.

The source comes from CUDA Samples commit
`5443602d89ed99aede2e4b7bf329daddeadb320e`,
`cpp/5_Domain_Specific/marchingCubes/marchingCubes_kernel.cu`, its `defines.h`
and the original `Common/helper_math.h` interpolation and cross-product helpers. Device function
bodies and copyright notices are retained; desktop host code is omitted.
`tables.json` contains the original `triTable` and `numVertsTable` from
`tables.h` (`X` is 255), under the same NVIDIA BSD-3-Clause license.
See the full notice in `kernel.cu` and the repository's third-party notices.

`pipeline.json` selects the existing upstream implicit-field branch through
compile definitions: `SAMPLE_VOLUME=0`, `USE_SHARED=1`, `NTHREADS=32` and
`SKIP_EMPTY_VOXELS=1`. `_DEFINES_H_=1` skips the embedded header's defaults.
These definitions override the defaults visible in the source editor; CUDA
function bodies are unchanged. The desktop host program is not executed. `volume-pipeline.json` selects
`SAMPLE_VOLUME=1` and the original shared `generateTriangles2` kernel, with
the original 32³ Bucky bytes in a normalized linear texture.

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

## Bucky sampled-volume profile

The separate Bucky showcase uses the same generic pipeline and mesh renderer.
Its default isovalue 0.5 generates 5,545 active voxels, 33,378 vertices and
11,126 triangles. Native comparisons also cover isovalues 0.2 (14,508 triangles)
and 0.8 (3,738 triangles). All prefix offsets and compacted indices agree exactly;
all positions and normals agree within 1e-6. The complete GPU pipeline loads no
native intermediate fixtures. It reads only 8 control bytes before final mesh
inspection. Native harness scans prepare expected captures on the host;
the browser computes its own scans on the GPU.

The source's `float3 *v[3]` is compiled as a local array of offsets into its
shared vertex allocation. Helpers specialize to that allocation, preserving
aliasing, pointer offsets and original `calcNormal` behaviour. There is no CPU
triangle or normal calculation. The original normal code produces per-triangle
normals, so this mesh is faceted. Disconnected small surfaces are also present
in the native result; the preview does not remove or smooth them.

The kernel's original output guard is `index < maxVerts - 3`. The native harness
allocates three extra records and supplies that capacity; the browser reserves
15 vertices per voxel, so both include the last complete triangle. The draw
count comes from the GPU scan total and excludes spare capacity. To change the
threshold, edit both `isoValue` keys in Launch settings and run again.

Bucky data is copied unchanged from the upstream marchingCubes sample and
retains NVIDIA's BSD-3-Clause license. The local helper `cross` is copied
unchanged from `helper_math.h`, including its copyright notice.

Run `node scripts/test-marching-volume-sandbox.mjs` after a static build for the
Bucky browser check. `tests/marching-volume-native.cu` regenerates three native
captures; `tests/marching-volume-gpu.js` compares the entire compute sequence at
all three thresholds. The current supported profiles use `USE_SHARED=1` and
`NTHREADS=32`; local-array pointer slots for `USE_SHARED=0` remain unsupported.
