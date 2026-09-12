# CUDA path tracer: native baseline

Target: Roger Allen's CUDA translation of *Ray Tracing in One Weekend*, final
chapter branch `ch12_where_next_cuda`, revision
`ab140b12d4923b75270831baabab5e4d4209f305`:
https://github.com/rogerallen/raytracinginoneweekendincuda/tree/ab140b12d4923b75270831baabab5e4d4209f305

## Verified native result

The original program compiles and runs on the local NVIDIA RTX 5080 with
CUDA 13.3 and MSVC. No source or header was modified. NVCC build flags were
`-O3 -std=c++17 -arch=native`; this changes the old Makefile's GPU target to
the installed hardware without changing the program.

The full original defaults were retained: 1200 x 800 pixels, 10 samples per
pixel, 488 spheres, and a maximum of 50 path bounces. The original timer
reported **0.635 seconds**. That timer includes per-pixel RNG initialization,
rendering and synchronization, but excludes scene construction and PPM file
output. It is one observed run, not a stable benchmark or a WebGPU comparison.

`pathtracer-native.png` is a lossless conversion of all 960,000 pixels from
the native PPM. The low sample count produces visible Monte Carlo noise; the
image was not denoised, recoloured, or rendered by a substitute implementation.
The manifest records source and image hashes. This is a native baseline,
**not a working browser showcase**.

## Current compiler findings

`scripts/probe-pathtracer.mjs` records the first actual errors. Loading
`main.cu` directly stops at its external includes. The compiler now parses plain
public value classes, constructors, array fields and const methods, lowering
them to records and device helpers. The isolated `vec3.h` probe now reaches
the host-only stream operators before stopping; the dependent ray, hit-record and sphere headers now compile with their
required headers supplied by the probe. Const unary signs, indexed reads and writes now compile. No original function bodies are rewritten by those probes.

The class-stage test compares 512 cases with **53,760 values matching native
CUDA exactly**, covering construction, copying, accessors, squared length,
unary signs, dynamic indexed reads/writes, compound indexed updates, all six compound vector/scalar operators, normalization, free binary operators, dot/cross products and assignment copy isolation. Native
compilation uses the complete original `vec3.h`, `ray.h`, `material.h` and `sphere.h`; the GPU fixture retains
all unchanged CUDA vec3 and ray definitions, omitting only includes, header guards and
host stream functions.
This is a focused language test, not support for the complete header or path
tracer. Const writes and resolved recursive class calls are rejected.

On the real NVIDIA WebGPU adapter, adding dynamic array indexing exposed an
incorrect shared value after copying a class containing an array. The emitter
now constructs independent aggregate fields explicitly for initialization and
assignment. Both copy cases match native CUDA; the precise backend cause has
not been isolated. The full regression run passes 197 GPU checks with no
software adapter requested, alongside 619 unit tests.

Writable indexing currently accepts the original `return field[index]` reference
accessor and lowers it to an lvalue into the original receiver. It does not
claim general reference-returning methods. Mutable void methods and compound
operators ending in `return *this` now use a reference to the receiver.
Out-of-class definitions must match their declared signature. Compound class
operators currently work as statements; mutable aliasing reference arguments remain rejected; read-only aliases
such as `dot(v, v)` are supported. The device-only probe explicitly omits host stream operators and
now compiles all remaining vec3 definitions. This does not resolve header
loading, host stream I/O or the scene class hierarchy.

The original ray class now compiles with nested vec3 members. Its origin,
direction, point evaluation and independent copies match native CUDA. Nested
members invoke default constructors before the containing constructor body;
classes without an explicit constructor receive one when nested members need
initialization. Recursive class storage and arrays of nested classes remain
unsupported.

Member initializer lists now initialize fields in declaration order, before
the body, including nested copies and explicit nested constructor calls. The
focused lambertian, metal and dielectric constructor fixtures retain their
original constructors and data fields but omit inheritance and scatter methods.
Their albedo copies, fuzz clamp and refractive indices match the complete
original native classes. This does not establish pointer or virtual dispatch
support and is not a browser material rendering result.

Source inspection identifies the following connected work:

1. Class values, constructors, member methods, `this`, operator overloads,
   and the three-float `vec3` storage layout.
2. Scene records containing material pointers and pointer-to-pointer parameters.
3. Virtual `hit` and `scatter` calls for the finite sphere/material hierarchy.
4. Device `new`/`delete` for scene construction and cleanup.
5. cuRAND state, initialization and sampling with the original random sequence.
6. Full scene rendering and image comparison, followed by sandbox integration.

The native CUDA version already uses a bounded loop for path bounces, so
recursive ray traversal is not the first blocker here. Preserve the full
final-chapter scene; do not replace it with an earlier chapter and present that
as support for this target. A future browser renderer must execute generated
WGSL from the upstream source, not display the native reference image.

## Reproduction on this PC

Clone the branch into `.local/raytracing-cuda`, then check out the pinned
revision above. From a Visual Studio x64 developer command prompt in the
repository root:

```
nvcc -O3 -std=c++17 -arch=native .local/raytracing-cuda/main.cu -o .local/nvidia-checks/pathtracer.exe
.local\nvidia-checks\pathtracer.exe > .local/pathtracer-native.ppm 2> reports/pathtracer-native.txt
python scripts/capture-pathtracer.py
node scripts/probe-pathtracer.mjs
```

`capture-pathtracer.py` checks the pinned revision and unmodified tracked
source, validates dimensions and every output colour, and writes a PNG plus
the provenance manifest. Native PPM and executable remain under `.local`.

## Sphere intersection stage

The original sphere class, hit record and fieldless abstract hitable interface
now compile unchanged after includes/guards are removed for the probe. A
concrete sphere call executes the original intersection method. Five cases
cover near root, far root, inside origin, miss and tangent; hit distances,
positions, normals and untouched miss records match native CUDA exactly.

This stage supports forward class declarations, nested values in plain structs,
matching implementations of a fieldless abstract interface and mutable record
output references. A bug passing a member of a referenced record to another
helper was fixed: the emitted pointer now addresses the member, not its parent.

The initial sphere stage retained typed null tokens for copying and equality;
its native test uses a null material pointer. The allocation stage below adds
concrete non-null objects. Fabricated addresses and pointer arithmetic remain
unsupported. There is no
virtual call through a base pointer yet. The test must not be described as
support for the complete scene, dynamic materials or a rendered showcase.

## Invocation-local allocation stage

Concrete `new Class(...)`, arrow member access, const/mutable method calls and
`delete` now use a bounded private pool with 1,024 slots per allocated class per
GPU invocation. Released slots can be reused; exhaustion returns a null token.
The compiler metadata explicitly marks the pool as invocation-local and not
persistent. Object pointers cannot currently escape through storage bindings.

Two allocated spheres produce distinct identities and matching intersections;
an allocated vec3 can be normalized, deleted and replaced. The CUDA native test
uses real device `new`/`delete` for the same operations. These tests establish
observable allocation behaviour within one invocation only, not a performance
comparison with CUDA's device heap.

The original create_world/render/free_world sequence still needs persistent
storage, pointer arrays, base-pointer virtual dispatch and cuRAND. No full scene
support or showcase is claimed by this allocation stage.
