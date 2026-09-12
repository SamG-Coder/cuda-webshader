# NVIDIA cubemap texture

[Open in sandbox](../../sandbox.html?example=cubemap).

The original `transformKernel` from NVIDIA cuda-samples revision `5443602d89ed99aede2e4b7bf329daddeadb320e`, `cpp/0_Introduction/simpleCubemapTexture/simpleCubemapTexture.cu`, is unchanged. It samples all six 64 by 64 faces through 3D direction coordinates, then negates each value.

Input values and sampling settings reproduce the original sample: ascending float32 values, normalized linear sampling, wrap addressing, and seamless filtering disabled. The compiler maps the dominant direction component to a face and face coordinates. Runtime storage uses six scalar float texture layers so sampling retains within-face address behavior. This is necessary for the original wrap settings: a clamp sampler passes center-only lookups but fails the captured edge probes.

The MIT `arrangeFaces` display helper places the original results in a 3 by 2 atlas: +X, -X, +Y across the top; -Y, +Z, -Z across the bottom. Grayscale maps -24575 to black and zero to white. The helper only rearranges output for display.

## Verification

All 24,576 WebGPU outputs match the native expected values exactly. Twelve additional native probe directions near the edges of all six faces also match exactly, including sampling through a device helper. All 12 native edge results change with seamless filtering enabled; that mode is currently rejected explicitly by the runtime.

The independent sandbox check verifies every original output, every atlas position and all 24,576 preview pixels, unchanged displayed source, and both generated shader passes. No software WebGPU adapter is used.

Run `node scripts/test-cubemap.mjs` for the native GPU comparisons. After `npm run build`, run `node scripts/test-cubemap-sandbox.mjs`. Set `CW_BASE_URL` to test deployment. Native reproduction and source references are in `reports/cubemap-progress.md`; capture hashes are in `reports/cubemap-native-manifest.json`.

The supported intrinsic is `texCubemap<float>` with nonseamless sampling. NVIDIA source retains BSD-3-Clause licensing. The compiler/runtime implementation, native probe harness, pipeline and display helper are MIT project code.
