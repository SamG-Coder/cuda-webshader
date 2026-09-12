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
