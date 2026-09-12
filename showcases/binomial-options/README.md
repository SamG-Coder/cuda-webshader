# NVIDIA binomial options

[Open in sandbox](../../sandbox.html?example=binomial).

Original device code from NVIDIA cuda-samples revision `5443602d89ed99aede2e4b7bf329daddeadb320e`, `cpp/5_Domain_Specific/binomialOptions/binomialOptions_kernel.cu`. CUDA helper and kernel bodies are unchanged. The standalone file supplies the original header constants (2048 time steps, 1024 maximum options) and default float precision, and omits desktop includes and host functions.

The complete default batch runs 1024 blocks of 128 threads. Every block evaluates one option through all 2048 time steps. Input structs come from the original native host preprocessing, captured after its CUDA symbol upload; each record retains five float32 fields and a 20-byte stride. Constant structs use read-only GPU storage, while the original `d_CallValue` device array uses persistent writable storage.

The preview is a 32 by 32 grid of the 1024 computed call values. Cells remain in original input order. Black means zero; white means 30. The display range is a visualization setting and does not modify kernel output.

## Verification

The original CUDA host wrapper and CPU reference run at full size using the original Windows input initialization (`srand(123)`). Native versus original CPU L1 error is 5.20082632e-08, within NVIDIA's original 5e-4 tolerance.

All 1024 WebGPU outputs are compared against native CUDA. The real NVIDIA run has relative L1 error 1.1014507031980355e-39 and maximum absolute error 1.0460491249205897e-37. These tiny differences occur among near-zero values; this is a tolerance comparison, not a claim of bitwise equivalence. The independent sandbox check verifies the complete output, original displayed source and generated shader selection. No software GPU fallback is used.

Run `node scripts/test-binomial.mjs` for the full real-GPU comparison. After `npm run build`, run `node scripts/test-binomial-sandbox.mjs`. Set `CW_BASE_URL` to the published site to test deployment. Native reproduction and capture hashes are in `reports/binomial-options-progress.md` and `reports/binomial-options-native-manifest.json`.

NVIDIA source retains BSD-3-Clause licensing. Project compiler, capture harness, pipeline and presentation code are MIT.
