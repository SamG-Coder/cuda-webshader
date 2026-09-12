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
