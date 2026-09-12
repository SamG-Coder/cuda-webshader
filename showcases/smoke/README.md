# NVIDIA smoke particles

[Open the sandbox](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=smoke).

The original noise3D, integrate_functor and calcDepth_functor bodies are retained
from NVIDIA cuda-samples revision 5443602d89ed99aede2e4b7bf329daddeadb320e,
cpp/5_Domain_Specific/smokeParticles. The CUDA/WGSL comparison includes these
compiled functions, an MIT GPU index initializer replacing host thrust::sequence,
and the runtime sorting helpers.

The host preset provides 16,384 particles in a deterministic sphere and the
original generated 64-cubed float4 noise texture. It uses NVIDIA's integration
defaults: timestep 0.5, damping 1, noise frequency 0.1, amplitude 0.001, zero
gravity and zero noise motion. It is a finite cloud, not the desktop emitter UI:
lifetime is 1000 simulation units and preview radius is 0.06. Particle positions
are never calculated on the CPU after loading.

The WebGPU renderer adapts NVIDIA's GLSL motion-stretched billboards and
half-angle slicing. Its 32 slices alternate camera accumulation and a 256-square
light attenuation buffer, with reversed camera blending when required.
Light position (5,5,-5), colour (1,1,0.5), sprite alpha 0.1, shadow alpha 0.005
and attenuation (0.1,0.2,0.3) follow the original defaults. This explains the
yellow light. Optional desktop blur and emitter controls are not implemented.
The camera uses OrbitControls; orbiting while paused recomputes depth ordering
without advancing integration.

Compute validation compares native Thrust/CUDA against the generated WebGPU
kernels, including the exact showcase inputs at step 64. Renderer validation
checks actual GPU output, shadow-on/off differences and both blending directions.
It is not a pixel-equivalence claim against the desktop OpenGL renderer or a
native/WebGPU performance comparison.

NVIDIA code and the adapted renderer are BSD-3-Clause; see
../../licenses/nvidia-cuda-samples-BSD-3-Clause.txt. Host/runtime adapters remain
MIT. Renderer WGSL is presentation code, not generated from the CUDA source.
