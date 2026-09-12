# NVIDIA particles: collision pipeline progress

Pinned source: `5443602d89ed99aede2e4b7bf329daddeadb320e`,
`cpp/2_Concepts_and_Techniques/particles`.

The original `calcHashD`, `reorderDataAndFindCellStartD`, `collideD`, grid helpers
and sphere/cell collision helpers now run through the compiler. Their device
function bodies remain unchanged in `tests/particle-collision-kernel.cuh`.
Host/library includes and the integration functor are outside this isolated fixture.

Compiler support added:
- Initialized local `volatile` value snapshots. Subsequent mutation, volatile
  pointers, shared memory and arrays remain rejected; this is not general
  volatile memory or synchronization support.
- `length` on float vectors, including the CPU oracle.

The runtime adds stable unsigned key/value sorting through compiled CUDA
bitonic passes, with no intermediate readback. It preserves input ordering for
equal keys and handles maximum uint keys, non-power-of-two counts and trailing
guards. GPU reference tests cover 1, 2, 127, 128, 129, 257 and 16,384 records.
This is a library replacement, not execution of Thrust in the browser.

Native tests use the original kernels, actual `thrust::sort_by_key`, 128- and
257-particle fixtures, and NVCC `-O3 --fmad=false -std=c++17 -arch=native`.
The fixtures deliberately contain nearby particles that collide, nonzero
velocities and a configured collider. They use 64³ cells and include a partial
final workgroup. Hashes, indices, complete cell ranges and reordered positions
and velocities match native exactly. Collision velocities differ by at most
4.656613e-10 (enforced absolute tolerance 1e-7). All pipeline stages exchange
GPU buffers; readbacks only validate their results.

## Complete simulation and showcase

The original `integrate_functor` body is now supported through a bounded zip
functor launch: one captured float constructor argument and two float4 tuple
elements accessed through `cuda::std::get<0/1>`. The parser supplies a guarded
thread launch and lowers tuple accesses; the original calculation is unchanged.
This is not general C++ functor or Thrust support. Nontrivial constructors,
additional state and other tuple layouts remain unsupported.

The native harness runs the original functor through real `thrust::for_each`,
then the original hash/reorder/collision kernels and Thrust sorting. The browser
runs the complete same sequence for 1,024 particles. Positions and velocities
are checked after steps 1, 8, 32 and 64. Maximum position error is 8.34465e-7
(tolerance 2e-6); maximum velocity error is 1.48721e-8 (tolerance 1e-7).
See `particle-simulation-check.json` and native captures. Initial positions and
collider placement are documented project fixtures, not the upstream UI defaults.

`example=particle-collision` is an animated sphere preview using shared GPU
position storage. Per-step integration, hashing, sorting, cell clearing and
collisions stay on the GPU. A labelled project CUDA clearing kernel replaces
the host's memset. Initial position readback frames the camera; subsequent
simulation has no readback. Three.js provides project sphere shading and a ground
grid. The code and generated WGSL remain available for comparison.

The sandbox test checks native values at step 64, automatic animation,
pause/resume, source preservation and mobile layout. It also checks finite
positions and cube bounds through step 256; this longer run is not a native
trajectory comparison. Floating-point collision trajectories can diverge over
longer runs. No performance parity with Thrust is claimed.
