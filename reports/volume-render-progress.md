# NVIDIA volumeRender: in progress

Target: the original full ray-marching kernel from NVIDIA CUDA Samples revision
`5443602d89ed99aede2e4b7bf329daddeadb320e`,
`cpp/5_Domain_Specific/volumeRender/volumeRender_kernel.cu`.

This candidate is not yet a working sandbox showcase. The source is not being
replaced by a custom JavaScript renderer or a simplified CUDA kernel.

## Verified prerequisite: ray vector math

The compiler now supports matching float-vector `dot`, `normalize`, vector
`fminf`/`fmaxf`, `make_float3(float4)`, `make_float4(float3, float)`, and named
float-vector compound assignments with `+ - * /` and a matching vector or float
scalar. These implement the used NVIDIA helper_math operations; arbitrary C++
operator overload declarations remain unsupported. Normalization validation uses
finite, nonzero vectors; behavior for degenerate/non-finite inputs is not claimed.

`tests/ray-vector-math.cu` is a project-owned probe, not an NVIDIA showcase kernel.
The native harness uses NVIDIA helper_math to check the same operations against
known geometric results. The WebGPU check uses two inputs uploaded to a GPU
buffer, including different directions and vector widths. All output components
must be finite and within 0.000002 of the independent expected values.

Evidence: `reports/ray-vector-math-native.txt`, the ray-math entry in
`reports/nvidia-regression-gpu.json`, and `tests/ray-vector-math.test.mjs`.

## Remaining work for the complete candidate

The current full-source importer/compiler probe fails at:

```
Unsupported type 'float3x4'. (53:14)
__constant__ float3x4 c_invViewMatrix;
```

Source inspection also identifies these required capabilities:

- Preserve and compile the `Ray` and `float3x4` struct declarations, including the
  matrix's float4 array and constant-memory inputs.
- Resolve the two overloaded `mul` helpers and their const struct/vector
  references.
- Pass addresses of local `tnear`/`tfar` values into `intersectBox`.
- Bind and sample the `tex1D<float4>` colour-transfer texture alongside the
  existing `tex3D<float>` volume texture.
- Configure the original camera matrix and transfer table, validate complete
  native and WebGPU ray-marched images, then expose the verified renderer as its
  own sandbox-linked showcase.

The vector tests do not prove the full volume renderer works. No showcase card
will claim that until the complete kernel and rendered output are verified.
