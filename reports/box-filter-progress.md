# NVIDIA boxFilter verification

The original six CUDA kernel entries compile. The colour column pass required
same-allocation pointer reassignment (`id = &id[x]`) and promotion of integer
scalars for float-vector arithmetic. No NVIDIA device function body was changed.
Pointer rebasing changes a lane-local offset, not the bound storage allocation.
The CPU oracle now copies pointer-parameter cells per lane while sharing data;
its previous shared cells made rebasing leak between simulated lanes.

Native CUDA captures cover 64x37 radii 0/3 and the original 1024x1024 teapot image
at radii 14/22. Both row and final colour images are compared, including guard
records. Radius 0 matches exactly; the other cases differ by at most one 8-bit
channel level. The four scalar global/texture row/column entries also match
native exactly on 64x32 data at radius 3, including guards. Native compilation
uses --fmad=false. These are correctness results, not performance comparisons.

The standalone showcase links directly to the sandbox. Its generic JSON pipeline
now supports PPM RGBA textures and packed image previews alongside triangle meshes.
The two colour passes share GPU buffers; no control or intermediate image readback
occurs. Only the final result is inspected for canvas display. Native fixtures are
expected results only. The preset executes one two-pass iteration, not the desktop
OpenGL application's full host loop.

Validation: 486 unit tests, 143 real NVIDIA hardware GPU checks, native captures,
compile-all, static build, the box-filter browser check at radii 14 and 22, and
all 28 built-in presets plus 38 imported NVIDIA entries pass. The browser check
compares every output channel and sampled canvas pixels, verifies both generated
passes and unchanged source, and checks invalid dimensions and mobile layout.
No software adapter was requested.
