# Third-party notices and license scope

The original CUDA WebShader source, kernel examples, generated shaders, tests,
benchmark harness, and project documentation are licensed under the MIT License
in [LICENSE](LICENSE). Third-party packages and external tools retain their own
licenses; the project's MIT license does not relicense them.

| Component | Version | License | Use |
|---|---|---|---|
| Three.js | 0.186.0 | MIT | Browser rendering dependency |
| Monaco Editor | 0.52.2 | MIT, with bundled third-party notices | Sandbox source editor used by VS Code |
| Playwright | 1.56.1 | Apache-2.0 | Development/browser test runner |
| playwright-core | 1.56.1 | Apache-2.0 | Transitive browser test dependency |
| fsevents | 2.3.2 | MIT | Optional macOS-only Playwright dependency |
| NVIDIA CUDA Samples simpleGL kernel | commit 5443602d89ed99aede2e4b7bf329daddeadb320e | BSD-3-Clause | Imported, unchanged showcase kernel |
| NVIDIA CUDA Samples N-body device code | commit 5443602d89ed99aede2e4b7bf329daddeadb320e | BSD-3-Clause | Original integration kernel, helpers and type declarations in the N-body showcase and tests |
| NVIDIA CUDA Samples FDTD3d kernel | commit 5443602d89ed99aede2e4b7bf329daddeadb320e | BSD-3-Clause | Original 3D finite-difference kernel and launch constants in the volume showcase and tests |
| NVIDIA CUDA Samples bicubicTexture | commit 5443602d89ed99aede2e4b7bf329daddeadb320e | BSD-3-Clause | Original render kernels and filter helpers in the validation fixture and bicubic-texture showcase; reuses the attributed simpleTexture teapot image |
| NVIDIA CUDA Samples convolutionTexture | commit 5443602d89ed99aede2e4b7bf329daddeadb320e | BSD-3-Clause | Original row/column kernels, unrolled helpers, IMAD macro and constant declarations in the validation fixture and convolution-texture showcase; reuses the attributed simpleTexture teapot image |
| NVIDIA CUDA Samples simpleSurfaceWrite | commit 5443602d89ed99aede2e4b7bf329daddeadb320e | BSD-3-Clause | Original surface-write and transform kernels, teapot512.pgm and derived validation images |
| NVIDIA CUDA Samples simpleTexture | commit 5443602d89ed99aede2e4b7bf329daddeadb320e | BSD-3-Clause | Original transformKernel, teapot512.pgm image and derived validation images |
| NVIDIA CUDA Samples volumeRender | commit 5443602d89ed99aede2e4b7bf329daddeadb320e | BSD-3-Clause | Original ray-marching device functions, camera declarations, Bucky volume and transfer table in the showcase and validation fixtures |
| NVIDIA CUDA Samples simpleTexture3D | commit 5443602d89ed99aede2e4b7bf329daddeadb320e | BSD-3-Clause | Original d_render kernel and Bucky.raw sample volume |
| NVIDIA CUDA Samples recursive Gaussian | commit 5443602d89ed99aede2e4b7bf329daddeadb320e | BSD-3-Clause | Original recursive filter, pixel conversion helpers and transpose; upstream credits CImg contributors |
| NVIDIA CUDA Samples Haar wavelet | commit 5443602d89ed99aede2e4b7bf329daddeadb320e | BSD-3-Clause | Original dwtHaar1D kernel and launch constants in showcase and tests |
| NVIDIA CUDA Samples separable convolution | commit 5443602d89ed99aede2e4b7bf329daddeadb320e | BSD-3-Clause | Original row/column kernels and constants in the two-pass showcase and tests |

Versions and package license declarations are recorded in package-lock.json.
Monaco's upstream LICENSE and ThirdPartyNotices.txt are included alongside its
redistributed browser assets in static builds. The sandbox serves Monaco locally
and does not upload pasted source to an editor service. Its C++ syntax highlighting
does not include the VS Code C++ extension's language server or native debugger.
Dependencies are installed separately with npm; node_modules is not committed.
Playwright's NOTICE credits Microsoft Corporation and code derived from
Puppeteer under Apache-2.0. Preserve the corresponding package LICENSE and
NOTICE files if redistributing those packages. Playwright is not included in
the browser application's static build.

Three.js is Copyright (c) 2010-2026 three.js authors. Its complete upstream MIT
license is preserved in [licenses/three-MIT.txt](licenses/three-MIT.txt).
The static build also includes node_modules/three/LICENSE alongside its
redistributed Three.js modules.

The CUDA Toolkit, NVIDIA driver/runtime, Visual Studio toolchain, and browser
executables are external prerequisites. Their binaries, SDK headers and
libraries are not distributed in this source repository. Native CUDA programs
are built locally using the user's installed toolchain. Their use and any
later redistribution of NVIDIA components remain subject to NVIDIA's terms.
The MIT license here covers the project's original CUDA source, not CUDA itself.

The implementation's design references are listed in [docs/research.md](docs/research.md).
The imported `showcases/simplegl/kernel.cu` retains NVIDIA's full copyright
and BSD-3-Clause notice; its generated WGSL is a translation of that licensed
source and retains that license. See [the upstream license](licenses/nvidia-cuda-samples-BSD-3-Clause.txt)
and [showcase provenance](showcases/simplegl/README.md). This is an explicit
exception to the original project code's MIT license. Standard CUDA algorithms and API use do not make the
project an NVIDIA SDK distribution. CUDA and NVIDIA names identify compatible
technology; no affiliation, endorsement or trademark rights are granted.

Primary license references:

The extracted NVIDIA kernels in `showcases/nvidia/kernels/` and their translated
WGSL in `showcases/nvidia/artifacts.json` are also BSD-3-Clause, not MIT. Each
CUDA file retains the original NVIDIA notice. The artifact manifest records
the upstream file and entry name; `reports/nvidia-audit.json` records the exact
upstream revision. The explorer, fixture harness and audit scripts are original
project code under MIT.

`tests/volume-transfer-kernel.cuh` retains NVIDIA's BSD-3-Clause notice and the
unchanged `d_integrate_trapezoidal` kernel from `volumeFiltering/volumeRender_kernel.cu`
at revision `5443602d89ed99aede2e4b7bf329daddeadb320e`. The float4 surface ABI probe
and native harness are project code under MIT.

The same upstream volume-rendering source supplies the unchanged functions in
`tests/volume-preintegrate-kernel.cuh`, `tests/volume-preintegrated-render-kernel.cuh`
and `showcases/volume-preintegrated/kernel.cu`. Both original transfer initializers
are retained in `tests/volume-transfer-colors.cuh`; their JSON values and the
showcase launch configuration include that NVIDIA data. These source/data portions
are BSD-3-Clause. The launch plumbing, validation code and icon are MIT.

- [MIT license text](https://opensource.org/license/mit)
- [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- [NVIDIA CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/index.html)

The `showcases/postprocess-gl/` CUDA source and `teapot.ppm` image are from NVIDIA CUDA Samples `postProcessGL`, revision `5443602d89ed99aede2e4b7bf329daddeadb320e`, under the NVIDIA BSD-3-Clause notice retained in `kernel.cu` and `licenses/`. The image is copied from `data/teapot_orig.ppm`. Original project integration and validation code remain MIT.

The CUDA code and `nature.bmp` in `showcases/bilateral-filter/` come from NVIDIA CUDA Samples `bilateralFilter` at revision `5443602d89ed99aede2e4b7bf329daddeadb320e`. The image is `data/nature_monte.bmp`; the code retains NVIDIA's BSD-3-Clause notice. Original compiler, decoder, integration and tests remain MIT.

The CUDA declarations in `showcases/mandelbrot/kernel.cu` and `tests/mandelbrot-kernel.cuh` are extracted from NVIDIA CUDA Samples `Mandelbrot/Mandelbrot_kernel.cuh` and `Mandelbrot/Mandelbrot_cuda.cu` at revision `5443602d89ed99aede2e4b7bf329daddeadb320e`. Original function bodies, NVIDIA BSD-3-Clause notice and DSFUN90 attribution comments are retained. Original compiler, integration, illustration and validation code remain MIT.

The VolumeTypeInfo conversion templates in `tests/volume-convert-kernel.cuh` are from NVIDIA CUDA Samples `volumeFiltering/volume.h` at revision `5443602d89ed99aede2e4b7bf329daddeadb320e`. Their original bodies and BSD-3-Clause notice are retained. The appended validation entries and native harness are original MIT code.

The original volume-filter kernel and conversion templates in `tests/volume-filter-kernel.cuh` are extracted from NVIDIA CUDA Samples `volumeFiltering/volume.h`, `volumeFilter.h` and `volumeFilter_kernel.cu` at revision `5443602d89ed99aede2e4b7bf329daddeadb320e`. Original device code and the BSD-3-Clause notice are retained. Validation hosts and GPU checks are original MIT code.

`showcases/volume-filter/kernel.cu` contains the same NVIDIA device declarations as `tests/volume-filter-kernel.cuh`. `showcases/volume-filter/Bucky.raw` is copied from NVIDIA CUDA Samples `volumeFiltering/data/Bucky.raw` at the same pinned revision. NVIDIA source and sample data remain under BSD-3-Clause; their notice is retained in the source and `licenses/nvidia-cuda-samples-BSD-3-Clause.txt`. Original sandbox integration, tests and `src/assets/volume-filter.svg` are MIT.

`tests/marching-interpolation.cuh` retains the original `vertexInterp2` helper from NVIDIA CUDA Samples `marchingCubes/marchingCubes_kernel.cu`, and `tests/marching-lerp.cuh` retains its scalar/vector interpolation dependencies from `Common/helper_math.h`, at revision `5443602d89ed99aede2e4b7bf329daddeadb320e`. Their original bodies and NVIDIA notices are preserved. Native/GPU validation wrappers are MIT.

`tests/marching-cubes-kernel.cuh` contains device declarations extracted from NVIDIA CUDA Samples `marchingCubes/defines.h` and `marchingCubes_kernel.cu`. `tests/marching-cubes-tables.cuh` and `tests/marching-bucky.raw` are the original `tables.h` and `data/Bucky.raw` from the same sample, revision `5443602d89ed99aede2e4b7bf329daddeadb320e`. NVIDIA code/data retain BSD-3-Clause licensing and source notices. Native and WebGPU classification hosts are MIT.

### Marching-cubes showcase assets

`showcases/marching-cubes/kernel.cu` retains the original device functions,
header declarations and interpolation helpers from NVIDIA CUDA Samples commit
`5443602d89ed99aede2e4b7bf329daddeadb320e`; its NVIDIA BSD-3-Clause notice is
included in the file. `showcases/marching-cubes/tables.json` reproduces
`triTable` and `numVertsTable` from the same sample's `tables.h` and is covered
by that same notice (also retained in `tests/marching-cubes-tables.cuh`).
The JSON launch configuration, generic sandbox pipeline, scan kernels and mesh
renderer are original MIT project code. Neither the MIT license nor the
translation into WGSL changes the NVIDIA source's BSD-3-Clause license.

`tests/marching-cross.cuh` and the added `cross` helper in the marching-cubes
showcase retain the original `Common/helper_math.h` NVIDIA code and notice.
`tests/shared-pointer.cu` retains the original NVIDIA `calcNormal` function and
notice; its other functions are MIT test code. `showcases/marching-cubes/Bucky.raw`
is the unmodified NVIDIA marchingCubes sample data, licensed under the same
BSD-3-Clause notice. The native volume captures derive from those assets.

### NVIDIA boxFilter

`showcases/box-filter/kernel.cu` and `tests/box-filter-kernel.cuh` retain the
original device functions and BSD-3-Clause notice from CUDA Samples commit
`5443602d89ed99aede2e4b7bf329daddeadb320e`,
`cpp/2_Concepts_and_Techniques/boxFilter/boxFilter_kernel.cu`.
`showcases/box-filter/teapot1024.ppm` is the sample's unchanged image under the
same NVIDIA license. Native captures derive from those assets. The compiler,
JSON launch setup and test harnesses are original MIT project code.

### NVIDIA SobelFilter

`tests/sobel-compute.cuh` retains the original `ComputeSobel` device function
and BSD-3-Clause notice from CUDA Samples commit
`5443602d89ed99aede2e4b7bf329daddeadb320e`,
`cpp/5_Domain_Specific/SobelFilter/SobelFilter_kernels.cu`.
The neighbourhood captures derive from this helper; wrappers, compiler support
and validation harnesses are original MIT project code.

`showcases/sobel/kernel.cu` and `tests/sobel-image-kernel.cuh` retain the same
NVIDIA notice and unchanged `ComputeSobel`, `SobelTex` and `SobelCopyImage`
functions. `showcases/sobel/teapot.pgm` is the original SobelFilter input under
that BSD-3-Clause license. Image captures derive from those sources and data.

`tests/sobel-shared-kernel.cuh` and `showcases/sobel/kernel.cu` additionally retain
the original NVIDIA `SobelShared` function and its BSD-3-Clause notice.

### NVIDIA imageDenoising

`showcases/denoising/kernel.cu` and `tests/denoising-kernel.cuh` retain device
functions, configuration expressions and BSD-3-Clause notices from CUDA Samples
commit `5443602d89ed99aede2e4b7bf329daddeadb320e`,
`cpp/2_Concepts_and_Techniques/imageDenoising/imageDenoising.cu`,
`imageDenoising.h` and the copy/knn/nlm/nlm2 kernel headers.
`showcases/denoising/portrait_noise.bmp` is the unchanged original sample asset
under the same NVIDIA license. The native captures derive from this source and
image. Host test harnesses and compiler support are original MIT project code.

### NVIDIA dct8x8

`tests/dct-float-kernel.cuh` retains the original first floating-point transform,
inverse transform, float quantization, constant matrices and BSD-3-Clause notices
from CUDA Samples commit `5443602d89ed99aede2e4b7bf329daddeadb320e`,
`cpp/2_Concepts_and_Techniques/dct8x8/dct8x8_kernel1.cuh` and
`dct8x8_kernel_quantization.cuh`. The default `FMUL` macro and block constants
come from that sample's `Common.h`. Native captures derive from these kernels.
The rounding fixture and test hosts are original MIT project code.

The DCT showcase also redistributes NVIDIA's original `dct8x8/data/teapot512.ppm`
and the selected kernel source in `showcases/dct/kernel.cu`, under the NVIDIA
BSD-3-Clause notice retained in that file and `licenses/`. The two explicitly
labelled pixel-conversion adapter kernels are project code under MIT.

The optimized DCT fixture and showcase retain the original device functions from
`dct8x8/dct8x8_kernel2.cuh` at the same pinned NVIDIA commit, under BSD-3-Clause.
Host-only includes are replaced with the required original block/multiply macros.
The source is in `tests/dct-optimized-kernel.cuh` and `showcases/dct-optimized/kernel.cu`.

### NVIDIA oceanFFT complete pipeline

`tests/ocean-kernel.cuh`, `tests/ocean-initial-spectrum.cuh`, and the upstream
portions of `showcases/ocean/kernel.cu` retain NVIDIA's BSD-3-Clause notices and
original device/host spectrum functions from `cpp/4_CUDA_Libraries/oceanFFT` at
commit `5443602d89ed99aede2e4b7bf329daddeadb320e`. `showcases/ocean/h0.bin` is
initial spectrum data generated by those host functions with seed 1. The GPU
inverse FFT, native test harness and labelled preview mesh adapter are project
code under MIT. The browser does not redistribute or execute cuFFT.

### NVIDIA particle collisions

`tests/particle-collision-kernel.cuh` retains NVIDIA's BSD-3-Clause notice,
`SimParams`, and the original grid, reorder and collision functions from
`particles/particles_kernel.cuh` and `particles/particles_kernel_impl.cuh` at
commit `5443602d89ed99aede2e4b7bf329daddeadb320e`. The runtime sorting kernels and
native validation harness are project code under MIT.

The particle simulation also retains the original `integrate_functor` definition
in `tests/particle-integration-kernel.cuh` and `showcases/particle-collision/kernel.cu`.
The original BSD-3-Clause notice remains intact. Compiler zip-launch lowering,
project initial-condition data and the labelled cell-clearing CUDA adapter are
project code under MIT.

`tests/smoke-noise-kernel.cuh` and `tests/smoke-noise-host.cuh` retain NVIDIA's BSD-3-Clause notice and the unchanged noise sampling and generation functions from `smokeParticles` at revision `5443602d89ed99aede2e4b7bf329daddeadb320e`. The surrounding probe and validation harness are MIT project code.

The smoke showcase includes original NVIDIA smokeParticles device functions and a WebGPU adaptation of SmokeShaders.cpp / SmokeRenderer.cpp in src/sandbox/smoke-renderer.js, under the NVIDIA BSD-3-Clause notice in licenses/nvidia-cuda-samples-BSD-3-Clause.txt.

### NVIDIA fluidsGL, stereoDisparity and HSOpticalFlow

The device functions in showcases/fluids, showcases/stereo and showcases/optical-flow, and corresponding tests, retain NVIDIA's BSD-3-Clause notices from cuda-samples revision 5443602d89ed99aede2e4b7bf329daddeadb320e. The relevant source directories are cpp/5_Domain_Specific/fluidsGL, stereoDisparity and HSOpticalFlow. The stereo packed-image inputs derive from stereo.im0.640x533.ppm and stereo.im1.640x533.ppm; the optical float-image inputs derive from frame10.ppm and frame11.ppm. These are NVIDIA sample assets, not new project imagery. The runtime adapters, host pipeline declarations and preview renderers are original MIT project code. See licenses/nvidia-cuda-samples-BSD-3-Clause.txt and each showcase README for provenance and input conversion.

The device functions in `showcases/fft-convolution/kernel.cu` and `tests/fft-convolution-kernels.cuh` come from NVIDIA cuda-samples `cpp/5_Domain_Specific/convolutionFFT2D/convolutionFFT2D.cuh` at revision `5443602d89ed99aede2e4b7bf329daddeadb320e`. They retain NVIDIA's BSD-3-Clause notice. The capture harness includes the original host wrappers and CPU reference from that sample; binary input data is generated with its original random seed. The host pipeline, FFT adapter and preview integration are original MIT project code.

`showcases/fft-convolution-custom/kernel.cu` combines unchanged padding, modulation and custom preprocessing device functions from the same pinned NVIDIA convolutionFFT2D sample and retains its BSD-3-Clause notice. The custom variant manifests and GPU copy integration are original MIT code.

The endpoint-fitting functions and constant tables in `tests/dxt-evaluate-kernels.cuh` come from NVIDIA cuda-samples `cpp/5_Domain_Specific/dxtc/dxtc.cu` at revision `5443602d89ed99aede2e4b7bf329daddeadb320e` and retain the BSD-3-Clause notice. Native DXT captures derive from the original teapot512_std.ppm sample asset. The test launch wrappers and compiler changes are MIT project code.

The original DXT functions and tables in `showcases/dxt/kernel.cu`, `tests/dxt-compress-kernels.cuh`, and `tests/dxt-colors-kernels.cuh` come from NVIDIA cuda-samples `cpp/5_Domain_Specific/dxtc` at revision `5443602d89ed99aede2e4b7bf329daddeadb320e`, under the retained BSD-3-Clause notice. The block-linear image input derives from NVIDIA's `teapot512_std.ppm`; the permutation input comes from its `permutations.h`. Native DDS and stage captures are test references. The appended display decoder and host pipeline declarations are original MIT project code.

`showcases/quasirandom/kernel.cu` and `tests/quasirandom-kernels.cuh` retain the original NVIDIA `quasirandomGeneratorKernel`, table declaration and macros from cuda-samples revision `5443602d89ed99aede2e4b7bf329daddeadb320e`, `cpp/5_Domain_Specific/quasirandomGenerator`. They retain the BSD-3-Clause notice. The direction table and native output captures derive from that sample's original CPU initializer and GPU kernel. The appended point-packing helper, pipeline and sprite renderer are original MIT project code.

`showcases/sobol/kernel.cu` and `tests/sobol-kernels.cuh` retain NVIDIA's `SobolQRNG/sobol_gpu.cu` body and BSD-3-Clause notice from cuda-samples revision `5443602d89ed99aede2e4b7bf329daddeadb320e`, including the credited contributions of Mike Giles, Frances Y. Kuo and Stephen Joe. Direction data derives from the original sample initializer and primitive table; native output captures are test references. The projection-packing helper and host pipeline are original MIT project code.
