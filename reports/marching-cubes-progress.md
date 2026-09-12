# NVIDIA marching cubes compiler work

Source: NVIDIA CUDA Samples `cpp/5_Domain_Specific/marchingCubes` at revision `5443602d89ed99aede2e4b7bf329daddeadb320e`.

The intended candidate is the original volume-to-triangle mesh pipeline. The current filter and volume-render showcases are separate; this candidate is not yet added as a working showcase.

## First blocker: linear texture fetch

Importing the original `defines.h` plus `marchingCubes_kernel.cu` initially failed parsing `tex1Dfetch<float>(volumeTex, i)`. The sample uses normalized byte fetches for volume values and unsigned integer fetches for vertex-count/triangle lookup tables.

The compiler now accepts `tex1Dfetch<float>` and `tex1Dfetch<uint>` with signed or unsigned integer indices. Texture type inference carries the distinction through device helper calls. Runtime `createLinearTexture` accepts Uint8Array normalized bytes or Uint32Array integer elements, packs long arrays across physical texture rows, and supplies the logical element count as a hidden u32 uniform. Uint textures use integer loads and cannot bind to float lookup parameters. Padding is excluded by the logical bounds check. Other element/read-mode combinations remain unsupported.

Validation uses original MIT probes in `tests/linear-texture-kernel.cuh`, compiled unchanged by CUDA and this compiler. Native CUDA uses `-O3 --fmad=false -std=c++17 -arch=native`. For 32,771 records and indices -2 through 32,772, all 65,550 results match exactly on the RTX 5080 (both integer results and normalized float results). This covers multiple physical rows, a padded final row, negative indices and indices beyond the logical length. Captures are retained in `reports/linear-texture-native-{float,uint}.bin` and checked by the hardware GPU suite.

458 unit tests and 133 hardware GPU checks pass. This proves the lookup feature for the stated profile; it does not prove marching-cubes classification or triangle generation yet.

## Next blockers from original source

After the lookup change, compile probes report:

- `classifyVoxel`: `uchar *volume` requires a packed byte-buffer ABI.
- `generateTriangles`: by-value `uint3 gridSize` is not accepted as a kernel argument.
- `generateTriangles2`: `uchar *volume` requires a packed byte-buffer ABI.

Further stages include the original scalar field/voxel classification, prefix scans and compaction, triangle interpolation/generation, and a generic mesh preview with native numeric comparisons. Existing upstream conditional paths and CUDA function bodies must remain intact. There is no new marching-cubes showcase card until its pipeline is actually verified.

## Vector launch values and read-only byte-pointer bindings

Kernel float/int/uint vectors of lengths 2, 3 and 4 now use named component uniforms (`parameter.x/y/z/w`), preserving signedness and validating each value before updating uniform memory. Like existing scalar parameters, these values are read-only in this compiler profile; writable local copies remain available. Three-component vector pointers remain rejected because their buffer layout is a separate concern.

Read-only `uchar*` kernel buffers use packed byte storage (one-byte logical stride, u32 physical words), with byte extraction at the translated address. Helper pointer offsets are applied in bytes, including offsets not aligned to a word. Writable byte pointers remain explicitly rejected. The sandbox seeds byte inputs as Uint8Array and can inspect their exact logical byte count. Local/shared byte values retain their existing physical layout accounting.

Native and hardware WebGPU agree exactly on 2,056 components combining seven vector launch values, signed values, unsigned wraparound and 259 packed input bytes read through shifted helper pointers. Unit coverage includes every float/int/uint vector length, invalid component ranges and transactional updates, and rejection of writes. A real Edge/NVIDIA sandbox check also verifies vector arguments, packed byte inputs and both integer/byte output inspection.

NVIDIA helper_math lerp definitions expose `inline __device__ __host__` qualifiers. The parser now accepts dual host/device helper declarations in either qualifier order while still rejecting host-only functions and host/global combinations. Their function bodies are unchanged.

The original marching-cubes source plus the original helper_math lerp functions now reaches `vertexInterp2`'s `float3&` output parameters. These require vector references to local values/array elements; that is the next compiler blocker. The complete mesh pipeline is still unverified and has no showcase card yet.

Validation: 464 unit tests, 134 NVIDIA hardware GPU checks, native captures, compile-all and static build pass. No software WebGPU checks were used.

## Local vector-reference outputs verified

Mutable helper references now accept exact-type numeric vectors and local array elements. Reference forwarding uses function pointers in WGSL; indexed arguments capture their index during argument evaluation. The CPU oracle preserves the selected element when another reference changes the index variable. Root-level alias rejection remains conservative: two references into the same array are rejected even when their indices differ. Storage, shared-memory, temporary and const output arguments remain rejected.

NVIDIA's original vertexInterp2 and original scalar/vector lerp bodies are retained in test fixtures with their notices. Native CUDA and hardware WebGPU agree exactly on 2,072 components (259 positions and 259 gradient vectors), including forwarded vector references and dynamic local-array slots. This validates the interpolation helper, not the entire mesh pipeline.

Original-source probes now reveal two distinct remaining blockers: classifyVoxel/generateTriangles2 use unsigned-vector addition for neighbouring grid positions, while generateTriangles's USE_SHARED branch passes shared-array elements as vector reference outputs. Shared-memory references require appropriate address-space handling; they have not been replaced with CPU work or silently switched to the local-memory branch.

Validation for this step: 468 unit tests, 135 hardware GPU checks, compile-all and static build pass. Native interpolation captures use multiply-add fusion disabled.
