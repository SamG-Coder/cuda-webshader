# NVIDIA volumeFiltering compiler progress
Candidate: CUDA Samples `cpp/5_Domain_Specific/volumeFiltering` at revision `5443602d89ed99aede2e4b7bf329daddeadb320e`.

The original `d_filter_surface3d` reads a 3D texture, applies constant float4 filter weights and writes a byte-valued 3D surface. Its source remains unchanged. The current sample is not running yet and has no showcase card.

## Completed prerequisite: 3D float surfaces
The compiler now infers 2D versus 3D surface bindings from intrinsic use. `surf3Dwrite` accepts float values, global XYZ coordinates, a byte X offset of globalX * 4, and default or explicit trap mode. It rejects mixed dimensions, shifted/unproven coordinates, other formats and other boundary modes. The runtime checks the entire dispatch against the surface dimensions before submission, including Z.

The runtime can allocate or upload r32float 3D storage textures. Existing r8unorm texture uploads retain their format and byte data. This step does not add float-3D sampling bindings or byte surface emulation.

An MIT probe executes the same CUDA function on native NVIDIA CUDA and hardware WebGPU: all 256 float voxels agree across eight workgroups. Hardware checks also reject oversized X, Y and Z dispatches. Native build: `nvcc -O3 -std=c++17 -arch=native tests/surface3d-native.cu -o .local/nvidia-checks/surface3d.exe`. Native result: `surface3d-native.txt`. GPU result: `nvidia-regression-gpu.json`.

## Remaining original-sample requirements
- Import and invoke static conversion templates: completed for explicit concrete type arguments; see validation below.
- Support the cudaExtent launch argument: completed for read-only by-value parameters with checked u32-range dimensions and unsigned 64-bit integer comparisons.
- Compile VolumeTypeInfo<unsigned char>::convert, including its original literal/cast arithmetic: completed and native/GPU verified.
- Support sizeof(VolumeType) with correct CUDA type semantics.
- Support byte-valued 3D surface writes and sampling the resulting volume without CPU round trips.
- Compare original filtering passes and final preview with native CUDA before adding a standalone sandbox showcase.

The float probe verifies a compiler/runtime prerequisite. It is not evidence that the original byte filter or complete desktop application works.

## Device declaration dependencies and value aliases
The importer now retains plain structs and scalar/vector typedef declarations only when referenced by extracted device declarations, following transitive dependencies and preserving source positions. Required unsupported structures remain visible and reject; they are not silently removed. The compiler supports unqualified built-in value aliases, chained aliases, scalar alias constructors/casts and aliases in explicit helper/kernel template arguments. Alias state is local to a parse and value-name shadowing rejects. Pointer/const aliases and class-method templates remain unsupported.

Native CUDA, the CPU oracle and real NVIDIA WebGPU agree on byte truncation, scalar conversion, helper template evaluation and packed component access. 433 unit tests and 126 NVIDIA hardware checks pass; compile-all and static build pass. No original volumeFiltering function body was changed, and the sample remains pending.

## Static conversion templates and exact byte arithmetic
The importer retains complete static template wrappers. The parser selects explicit concrete type specializations, lowers their unchanged methods to device helper ASTs, and supports primary methods as a fallback. Empty primary templates and unused static const metadata declarations are retained; static data access, instances, inheritance, member templates, dependent static-call type arguments, unqualified static member references and unsupported selected types remain rejected. Unselected unsigned-short methods do not enable short storage or arithmetic.

The original unsigned-char conversion uses a double literal in `__saturatef(sampled) * 255.0`. The compiler now handles a byte cast of a float32 operand multiplied by a positive integral double literal from 1 through 65535 using integer significand limbs. This preserves the exact product and truncation within the defined CUDA byte-conversion range without enabling general double precision. Out-of-range/NaN float-to-byte conversions are not a validated contract.

767 original NVIDIA conversions match native CUDA and hardware WebGPU, including adjacent float32 values at byte boundaries. A separate MIT arithmetic probe verifies 255 boundaries using multiplier 65535.0; all would differ if the product were rounded to float32 before conversion. All four original VolumeTypeInfo template definitions match the pinned header byte for byte.

At this stage the original filter reached its cudaExtent launch parameter. 441 unit tests and 128 NVIDIA hardware checks pass; compile-all and static build pass. No showcase is added until the original filter pipeline is verified.

## Extent parameters and vector filter weights
Read-only by-value cudaExtent parameters are now transported as three named u32 uniforms: `size.width`, `size.height`, `size.depth` for a parameter called size. Each component must be an integer from 0 through 4294967295; larger values are rejected, not truncated. All six integer comparison operators preserve native 64-bit unsigned size_t promotion, including sign extension when a negative int is compared with an extent field. This is a bounded extent-parameter profile, not general size_t arithmetic or arbitrary struct ABI support. Extent mutation, local extent values, extent pointers and arithmetic on size fields remain unsupported.

A native/GPU probe verifies 168 comparison results across dimensions 0, 8 and UINT32_MAX, including INT32_MIN and -1 coordinates, unsigned casts and cross-field comparisons. Invalid component updates reject transactionally before touching uniform memory.

Constant scalar/vector values and fixed vector arrays now use flattened uniform components. Vector arrays support 1..256 records with zero initialization and host overrides; existing struct component limits remain unchanged. The original filter’s `float4 c_filterData[125]` gets through compilation. A native/GPU probe checks all 125 float4 records (500 uploaded components) through helper reads.

446 unit tests and 130 NVIDIA hardware checks pass, plus compile-all and static build. The current original sample failure is `sizeof(VolumeType)` in the surf3Dwrite byte offset. Byte-valued 3D surface output and GPU-only sampling/rendering of that output also remain to be implemented and verified before a showcase can be added.
