# Native CUDA transport provenance

`transport.js` derives from `rtx_cuda/js/runtime.js` in
[ChromiumRTXCuda](https://github.com/SamG-Coder/ChromiumRTXCuda/blob/533fe7ef6c48871e0b082235268e83eb3eb0c3cd/rtx_cuda/js/runtime.js),
commit `533fe7ef6c48871e0b082235268e83eb3eb0c3cd`.
Copyright 2026 The ChromiumRTXCuda Authors. Distributed under the included
BSD-3-Clause LICENSE. The RTX/DLSS classes are omitted; this integration selects
CUDA compute or WebGPU and does not claim renderer interoperability.
The parent native-runtime.js adapter supplies selection metadata, artifact
support, validation, error reporting and the document-session ownership guard.
