# Native desktop showcase

The complete upstream NVIDIA simpleGL desktop sample was built locally from
commit 5443602d89ed99aede2e4b7bf329daddeadb320e with NVCC 13.3 and its original
CUDA/OpenGL/GLUT host code. Downloaded dependencies and binaries are kept in
.local/cuda-samples (ignored by Git).

Launch from PowerShell:

```powershell
Start-Process -FilePath "$PWD\.local\cuda-samples\bin\win64\Release\simpleGL.exe" -WorkingDirectory "$PWD\.local\cuda-samples\bin\win64\Release"
```

The application created a responding window titled "Cuda GL Interop (VBO)".
Drag with the left button to rotate; drag with the right button to zoom;
Esc closes the window. This is actual native CUDA/OpenGL execution. The sample's
FPS title is its own timing measurement and is not used for our cross-API
benchmark claims. See reports/simplegl-desktop-build.log for compilation evidence.
