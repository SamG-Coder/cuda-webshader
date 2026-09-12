# NVIDIA custom FFT convolution variants

[Custom FFT (test1)](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=fft-custom) · [Fused FFT (test2)](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=fft-fused)

Both original variants run on the GPU at NVIDIA's original 2000 × 2000 input size, using a 7 × 6 filter centred at (4,3), padded 2048 × 2048 real storage and 1024 × 2048 complex transforms. They reuse the original Windows srand(2010), rand()%16 captures from the default FFT convolution showcase.

Test1 performs forward complex FFTs, original spPostprocess2D_kernel conversion to padded real spectra, original modulation/normalization, spPreprocess2D_kernel conversion back, then inverse complex FFT. The original spectrum padding is 16 complex records per row.

Test2 performs forward complex FFTs, then original spProcess2D_kernel, which fuses both conversions with modulation. Its output uses a separate allocation to satisfy WebGPU's prohibition on overlapping writable storage bindings; source calculations and texture reads remain unchanged. An inverse complex FFT produces the final output.

Explicit byte copies reinterpret pairs of real floats as complex values without changing their bits. GPU buffer-to-linear-texture copies preserve logical indexing across physical texture rows and partial tails. No intermediate CPU readback occurs. The pipeline keeps the 128 MiB declared-buffer and 64 MiB texture budgets; FFT scratch buffers are additional runtime allocations. Test1 declares 117,187,752 buffer bytes; test2 declares 99,886,248. Runtime FFT helpers and host wiring are MIT project code; browser execution does not use cuFFT.

Every CUDA device function body in kernel.cu matches the pinned NVIDIA source. The comparison menu exposes original preprocessing kernels and generated FFT stages. The preview maps raw output to grayscale [0,5055], including the last 48 padding rows and columns. This is the original random numerical field, so its filtered image looks like noise. Display mapping does not modify the kernel output.

## Verification

All output components are compared with native CUDA at both 37 × 19 and original 2000 × 2000 input sizes. At full size:

| Path | Native relative L2 | Maximum native absolute error | Original CPU relative L2 |
|---|---:|---:|---:|
| Custom test1 | 2.02473e-7 | 0.0035400390625 | 1.40741e-7 |
| Fused test2 | 2.02391e-7 | 0.00341796875 | 1.40382e-7 |

The independent spatial convolution checks all four million input-domain pixels, including clamped borders, against NVIDIA's original 1e-6 relative L2 criterion. Both previews retain 182 grayscale levels. Hardware browser tests check source preservation, shader stages, mobile layout and switching examples. Software adapters are not used.

Native capture: tests/fft-custom-full-native.cu, with the pinned convolutionFFT2D source directory and Common on the include path, linked with cuFFT. Run node scripts/test-fft-custom-full.mjs for all complete GPU comparisons. After npm run build, run node scripts/test-fft-custom-sandbox.mjs with CW_FFT_VARIANT=1 or 2. CW_PUBLIC_URL can select the deployed preset URL.

Reports: reports/fft-custom-full-check.json, fft-custom-full-native.txt, fft-custom-sandbox-check.json and fft-fused-sandbox-check.json. Lower-level bidirectional stage tests remain in fft-custom-check.json.

Source: NVIDIA cuda-samples revision 5443602d89ed99aede2e4b7bf329daddeadb320e, cpp/5_Domain_Specific/convolutionFFT2D. The original BSD-3-Clause notice is retained in kernel.cu. The pipeline manifests, generic copy/FFT integration and preview are MIT project code.
