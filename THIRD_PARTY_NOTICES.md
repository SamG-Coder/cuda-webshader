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
