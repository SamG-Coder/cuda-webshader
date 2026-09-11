# Third-party notices and license scope

The original CUDA WebShader source, kernel examples, generated shaders, tests,
benchmark harness, and project documentation are licensed under the MIT License
in [LICENSE](LICENSE). Third-party packages and external tools retain their own
licenses; the project's MIT license does not relicense them.

| Component | Version | License | Use |
|---|---|---|---|
| Three.js | 0.186.0 | MIT | Browser rendering dependency |
| Playwright | 1.56.1 | Apache-2.0 | Development/browser test runner |
| playwright-core | 1.56.1 | Apache-2.0 | Transitive browser test dependency |
| fsevents | 2.3.2 | MIT | Optional macOS-only Playwright dependency |

Versions and package license declarations are recorded in package-lock.json.
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
The source review found no embedded third-party implementation bearing a
conflicting license. Standard CUDA algorithms and API use do not make the
project an NVIDIA SDK distribution. CUDA and NVIDIA names identify compatible
technology; no affiliation, endorsement or trademark rights are granted.

Primary license references:

- [MIT license text](https://opensource.org/license/mit)
- [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- [NVIDIA CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/index.html)
