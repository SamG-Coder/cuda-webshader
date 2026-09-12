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

## Verified prerequisite: local structs and constant camera matrices

The importer preserves plain named structs and anonymous struct typedefs. The
compiler supports flat scalar/vector fields and one-dimensional fixed field
arrays, local struct copies, and by-value helper parameters/returns. Struct
storage buffers, shared structs, nested structs and arrays of structs remain
unsupported: this does not claim binary compatibility between CUDA and WGSL
struct layouts. Each struct has at most 64 fields; field arrays have at most 256
elements. Local typed values are emitted as WGSL structs.

A single zero-initialized constant struct can expose up to 256 float/int/uint
components through existing uniform scalars. For example, the matrix component
uses the key `constant.camera.m[0].x`. Constant aggregate initializers are not
yet supported. The runtime retains scalar range validation and transactional
uniform updates. Assigning the constant to a local matrix creates a value copy.

Native CUDA and NVIDIA WebGPU independently check all three dynamic matrix rows,
helper return-by-value, original ray preservation, local matrix mutation and
unchanged constant contents. See `reports/local-structs-native.txt`,
`tests/local-structs.test.mjs` and the local-structs GPU regression entry.

## Remaining work for the complete candidate

The current full-source importer/compiler probe fails at:

```
Expected ')', found ','. (169:47)
float4 col = tex1D<float4>(transferTex, (sample - transferOffset) * transferScale);
```

Source inspection also identifies these required capabilities:

- Resolve the two overloaded `mul` helpers and their const struct/vector
  references.
- Pass addresses of local `tnear`/`tfar` values into `intersectBox`.
- Bind and sample the `tex1D<float4>` colour-transfer texture alongside the
  existing `tex3D<float>` volume texture.
- Configure the original camera matrix and transfer table, validate complete
  native and WebGPU ray-marched images, then expose the verified renderer as its
  own sandbox-linked showcase.

The prerequisite tests do not prove the full volume renderer works. No showcase card
will claim that until the complete kernel and rendered output are verified.
