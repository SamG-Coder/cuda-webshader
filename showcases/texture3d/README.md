# NVIDIA 3D texture slice

[Open the texture sandbox](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=texture3d).

The original `d_render` kernel and 32 × 32 × 32 `Bucky.raw` volume come from
NVIDIA CUDA Samples revision `5443602d89ed99aede2e4b7bf329daddeadb320e`, in
`cpp/0_Introduction/simpleTexture3D`. The kernel body is unchanged and retains
its BSD-3-Clause notice. Desktop allocation and window code are replaced by the
sandbox's launch and resource setup.

This displays a 128 × 128 slice through the volume. In Launch settings, change
`scalars.w` to choose its depth, or `textures.texObj.filter` between `linear`
and `nearest`, then Compile & run. Coordinates wrap, matching the original
sample's sampler settings. The view is a slice, not a ray-marched surface.

The compiler maps a by-value `cudaTextureObject_t` kernel parameter to a WebGPU
3D texture and sampler. `tex3D<float>` becomes `textureSampleLevel` at level zero.
The runtime allocates a real `r8unorm` 3D texture, uploads the byte volume once,
and binds the sampler. GPU hardware performs normalization and interpolation;
there is no JavaScript sampling in the application's execution path.

The current texture contract is unsigned 8-bit normalized scalar volumes,
normalized coordinates, nearest or linear filtering, and repeat, clamp-to-edge
or mirror-repeat addressing. Texture parameters in device helpers and additional
sandbox passes, other texture formats, and CUDA surfaces are not yet supported.
The CPU test interpreter does not provide a texture execution fallback.

The sandbox's `textures` object specifies each parameter's `dimensions`,
same-origin raw byte `source`, `filter` and `addressMode`. Dimensions must match
the file's byte count. Texture and buffer allocations share the sandbox's 64 MiB
limit. `image.format: "r8-in-u32"` displays the kernel's 0–255 uint output as
opaque grayscale without modifying the output values.

Native CUDA and NVIDIA WebGPU validate the unchanged kernel against an
independent 3D sampling reference using a patterned 8 × 4 × 4 volume, both
filters and four depths, including negative and greater-than-one coordinates.
A 17 × 9 output checks partial blocks; output guards must remain exact. Linear
filtering allows two 8-bit levels of error, nearest filtering one, accounting for
texture interpolation and float-to-integer rounding differences. Native observed
errors are zero for nearest and at most one for linear in these cases.

The built sandbox test additionally checks every displayed pixel from the
original Bucky volume, its NVIDIA adapter, final-only readback and mobile layout.
See `reports/texture3d-native.txt`, `reports/nvidia-regression-gpu.json` and
`reports/texture3d-sandbox-check.json`. These validate correctness and integration;
they are not CUDA-versus-WebGPU performance measurements.
