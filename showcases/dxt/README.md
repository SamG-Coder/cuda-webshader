# NVIDIA DXT texture compression

[Run in sandbox](../../sandbox.html?example=dxt).

Original `dxtc` device functions and tables from NVIDIA cuda-samples revision `5443602d89ed99aede2e4b7bf329daddeadb320e`, `cpp/5_Domain_Specific/dxtc/dxtc.cu` and `CudaMath.h`. The CUDA function bodies remain unchanged and retain BSD-3-Clause licensing. The final `decodeDxtPreview` function is an original MIT display helper, explicitly separated in the source.

The original 512 × 512 `teapot512_std.ppm` is stored in `input.bin` as block-linear RGBA uints, matching the original host program's input conversion. `permutations.bin` captures all 1024 original permutation words. Host declarations are in `pipeline.json`; no native output is used as browser input.

The compressor launches all 16,384 blocks with 64 threads each. The compiler lowers the first 16-thread colour tile into predicated workgroup phases with unconditional synchronization. This is bounded first-tile support, not arbitrary CUDA cooperative groups. Compressed uint2 blocks stay on the GPU for the display decoder; only the final RGBA image is read for presentation.

Verification on NVIDIA Blackwell: all 32,768 compressed words match the unchanged native CUDA program. All 262,144 decoded pixels match an independent JavaScript decoder of the native DDS, including transparent DXT1 blocks. See `reports/dxt-compress-check.json` and `reports/dxt-sandbox-check.json`. The original native sample also passes its own reference-image tolerance (RMS 0.007950, threshold 0.02).

Run `npm run build`, then `node scripts/test-dxt-sandbox.mjs` with real hardware Edge installed. The full compression and colour-stage native checks are also part of the hardware GPU suite.
