# NVIDIA volumeRender: complete device pipeline

Target: the original full ray-marching kernel from NVIDIA CUDA Samples revision
`5443602d89ed99aede2e4b7bf329daddeadb320e`,
`cpp/5_Domain_Specific/volumeRender/volumeRender_kernel.cu`.

The complete original device pipeline runs in the sandbox at `sandbox.html?example=volume`.
The CUDA function bodies remain unchanged; the browser supplies resources and launch settings.

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

## Verified prerequisite: float4 transfer textures

The compiler accepts `tex1D<float4>` for a by-value texture parameter, alongside
`tex3D<float>` in the same kernel. A 1D transfer table uses a one-row 2D
`rgba32float` WebGPU texture with normalized coordinates and level-zero
sampling. The runtime requests `float32-filterable` when available and reports
a clear error if a supplied device lacks it; the implementation does not reduce
the table to half precision. Nearest and linear filtering are supported. A
single handle cannot mix 1D and 3D sampling.

Sandbox transfer settings use `dimensions: [width]`, a flat `values` array
with four finite numbers per record, plus `filter` and `addressMode`. Texture
lifetimes and data uploads are managed by the runtime. Helper texture parameters
and additional-pass texture bindings remain unsupported.

Native CUDA and hardware WebGPU probes combine both texture dimensions in one
kernel and compare both filtering modes, clamped coordinates and shifted lookup
positions against an independent interpolation reference. The absolute tolerance
is 0.005 per float component, allowing CUDA texture interpolation quantization.
The native observed maximum was approximately 0.0019455; nearest cases matched
exactly. The built sandbox also loads both resource types and verifies its output.
See `reports/transfer-texture-native.txt`,
`reports/transfer-texture-sandbox-check.json` and the paired-texture GPU check.

## Verified prerequisite: transformation overloads and const references

Non-template device helpers can overload by parameter type. Calls require one
exact type match; implicit conversion ranking and ambiguous calls are rejected.
Duplicate signatures, kernel overloads and mixed template overload sets are not
supported. Internal names keep generated WGSL functions distinct.

Const references accept exact scalar/vector/local-struct types, including
constant matrices and temporary vector values. Function-addressable local values
use pointers; immutable values and temporaries receive local storage for the
call. Aliased reference arguments and references directly to storage-buffer
components are rejected. Existing mutable references remain limited to named
local numeric scalars.

The unchanged NVIDIA float3 and float4 `mul` helpers were copied with their
license into `tests/volume-mul.cuh`. A project-owned validation entry checks
that directions omit translation while points include it. Native CUDA, the typed
interpreter and NVIDIA WebGPU all produce the exact expected eight components
for a non-identity matrix and buffer-supplied point. See
`reports/volume-mul-native.txt` and the matrix-overload GPU regression entry.

## Complete ray-marching validation

Local scalar output pointers now support the original intersectBox call with
&tnear and &tfar. Typed specialization emits function-address-space pointers.
Only dereference or index zero is supported for these local pointer arguments;
pointer arithmetic, escape, mismatched pointee types and aliased arguments are
rejected. See tests/local-pointer.test.mjs.

The native harness runs the original device functions against NVIDIA's original
32³ Bucky volume and nine-entry transfer table. The WebGPU regression compares
all RGBA channels for 128×128 and 65×49 images, plus a rotated 128×128 camera.
Sixteen trailing output guards must remain intact in each case. Observed maximum
channel differences were 1, 0 and 1 on an 8-bit scale; the regression allows 2.
See reports/volume-render-native.txt, the three native binary images, and the
complete-volume-renderer check in reports/nvidia-regression-gpu.json.

The built sandbox test compares every displayed RGB pixel with native CUDA,
checks opaque presentation, source/WGSL comparison and mobile width. Presentation
uses opaque alpha to display the original premultiplied RGB without multiplying
it again through canvas compositing. The computed RGBA buffer is unchanged.

This verifies the complete device renderer, not the original desktop OpenGL
application or a CUDA-versus-WebGPU performance advantage. Camera, density and
transfer settings are editable JSON; rendering is recomputed on Compile & Run.
