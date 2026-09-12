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

The class-stage test compares 512 cases with **56,320 values matching native
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
not been isolated. The full regression run passes 198 GPU checks with no
software adapter requested, alongside 621 unit tests.

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
unsupported. The later dispatch stage below adds virtual calls through a fieldless base
interface. The test must not be described as
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
storage, pointer arrays and cuRAND. No full scene
support or showcase is claimed by this allocation stage.

## Tagged virtual dispatch stage

Object tokens now contain a concrete type tag and a pool slot. Converting a
concrete pointer to its fieldless abstract base preserves that identity. A call
through the base selects the matching allocated implementation at runtime;
delete through that base releases the corresponding pool slot.

The native/GPU fixture calls the original sphere hit method through hitable*.
A separate shared MIT fixture provides two different implementations of one
interface, selected by thread index, so dispatch is tested across concrete
types rather than only a single sphere implementation. All output values match
native CUDA exactly. Source bodies of the original sphere remain unchanged.

This supports the existing single, fieldless abstract base model and remains
invocation-local. Persistent world storage, pointer arrays and cuRAND still
block the full create_world/render/free_world pipeline and showcase.

## Persistent arena stage

Compile the complete kernel module with `objectHeap: 'persistent'`, create an
arena with `runtime.createObjectArena()`, and bind each kernel with the same
`{objectArena: arena}` third argument. Kernels share typed storage pools in
bind group 1. Atomic slot reservation supports concurrent creation; deletion
releases slots. Class-pointer buffer parameters (`Class**`) carry 32-bit
compiler object tokens, not native CUDA addresses. Pointer buffers are owned by
one arena, and incompatible arena layouts/type tags are rejected.

The original sphere/ray implementations now run in a three-kernel harness:
create 512 objects, trace in a separate submission, then delete and clear the
pointer buffer. Two complete cycles with different positions match all 5,120
native CUDA values exactly. All identities are distinct/non-null after creation
and every pointer is null after freeing. This is additional to the 56,320-value
invocation-local fixture. The CPU oracle explicitly rejects persistent arena
execution rather than silently resetting object state between dispatches.

Persistent fields currently require host-shareable values; mutable helper
references directly into persistent objects are rejected. The full original
world still needs its pointer-list object, camera and cuRAND integration. The
three-kernel harness is a verification stage, not the full path tracer or a
new showcase.


## Captured object-list stage

The original hitable_list constructor and hit traversal now execute with
persistent objects. A captured Class** buffer becomes a bounded buffer/offset
descriptor; the arena retains the actual GPU buffer across kernel submissions.
Imported buffers cannot be rebound or shared with another arena. Indexed reads
resolve the descriptor to the retained buffer. This stage supports direct
constructor captures and tracks directly allocated target types for dispatch;
it does not establish support for arbitrary recursive scene graphs.

The native CUDA and real hardware WebGPU harness create two spheres and a list,
trace a hit and miss, replace the nearer sphere in a later submission, and trace
again. All eight output values match exactly, including changed closest-hit
distance and the untouched miss record. Cleanup clears every pointer. The GPU
trace binds only its output; the arena retains both scene buffers. The original
list and sphere function bodies are unchanged. Camera, cuRAND and complete
material scattering still need integration before the full scene can be shown.


## Original camera and XORWOW stage

The complete original camera.h device definitions now compile and execute,
including random_in_unit_disk and non-const get_ray. The parser accepts grouped
class fields, mutable methods returning values, and unambiguous numeric
constructor conversions. Local record pointers can be forwarded through device
helpers and class methods without losing state updates. CUDA tan lowers to WGSL
tan under the existing float32 arithmetic model.

An explicit `libraries: ['curand-xorwow']` compilation option supplies a small
compatibility implementation of curandState, curand_init, curand and
curand_uniform. It supports uint32 seeds and compile-time zero subsequence and
offset, exactly the initialization mode used by the original final chapter.
Unsupported initialization modes are compilation errors. This is not the whole
cuRAND API, its native state ABI, or support for 64-bit seeds. The compiler-owned
state keeps the six XORWOW words needed by these operations. The original CUDA
program and native harness use the installed real cuRAND library.

Across 512 cameras (varying field of view, image coordinates and aperture), all
4,608 RNG integers match native CUDA exactly, including a draw after lens
sampling that verifies the state consumed by rejection sampling. All 4,096
uniform float values match exactly. The total 11,776 float outputs additionally
cover ray origins, ray directions and camera basis vectors; maximum absolute
error is 0.000003814697265625 (tolerance 0.00002 for camera arithmetic).
Seeds include zero and UINT32_MAX. Every original camera method body is intact.
These results establish camera and RNG behaviour in a local-value harness;
persistent camera access, material scattering and the full scene still require
integration before a new sandbox showcase is valid.


## Original material-scattering stage

The original material.h device definitions now execute, including RANDVEC3,
random_in_unit_sphere, reflect/refract, Schlick reflectance, and the lambertian,
metal and dielectric scatter methods. Bounded object-like primary-expression
macros now preserve repeated function calls and state effects. Unambiguous
constructors accept double-to-float scalar conversion, and CUDA float pow maps
to WGSL pow. Source method bodies and the original RANDVEC3 macro are unchanged.

The native harness uses real device allocation, material* virtual dispatch and
cuRAND. The WebGPU harness runs the same 512 cases in invocation-local and
persistent arena modes: 1,024 cases and 10,240 compared float values. All 1,024
subsequent RNG draws match the native values exactly. Scatter acceptance,
attenuation and ray origins match exactly; maximum absolute direction error is
0.00000011920928955078125 (tolerance 0.00003). Inputs include inward/outward rays,
multiple incident angles and metal fuzz values above one to exercise clamping.
The native cases include both accepted and rejected metal scatters.

This verifies the full original material methods and random sampling, not yet
the complete rendered scene. Combining all original device functions currently
stops at the explicit sphere* downcast in free_world. Persistent camera access,
class-value output buffers, shared parameter naming and RNG buffer initialization
also still need full-module integration. No new showcase card is claimed here.


## Record buffers, wide seeds and cleanup integration

Explicit casts from a fieldless base pointer back to its related concrete class
now preserve object identity. The original free_world body parses and compiles.
A separate native/WebGPU harness allocates 512 original sphere/lambertian pairs,
uses the original downcast-and-delete expression, and checks that every pointer
is cleared and all persistent allocation slots are free.

Host-shareable record buffers expose their WGSL-aligned record stride. An
explicit `valueBuffers: ['fb']` option interprets the original vec3* framebuffer
parameter as a value buffer; it does not reinterpret arbitrary object tokens.
The path tracer's compiler-owned XORWOW records occupy 24 bytes, and its vec3
records occupy 12 bytes. These are explicit WebGPU layouts, not a claim that all
CUDA C++ structs share the same ABI. The GPU test exposed and fixed an offset
being incorrectly applied both to the record and to its nested fixed array.

XORWOW initialization now accepts 64-bit integer seeds, including signed integer
conversion, while retaining the compile-time zero subsequence/offset restriction.
The original render_init kernel runs unchanged. Three passes over 512 state
records verify both persistent RNG advancement and seeds with nonzero high words
or negative signed inputs. All 4,608 float framebuffer components and 1,536
integer draws match native CUDA exactly. Both storage-buffer helper writes and
class-value output stores are exercised on the real NVIDIA adapter.

The complete device-module probe now compiles rand_init, render_init and
free_world. create_world next stops at increment inside an assignment destination;
render stops at the non-const camera method on a persistent object. These are
integration blockers still to resolve before the full scene and sandbox card.

## Full scene integration

All five original device entries now compile and run: rand_init, create_world, render_init, render and free_world. Indexed assignment preserves C++17 RHS-before-LHS sequencing; persistent member methods use storage-backed object indices; virtual calls forward local RNG references. The generic sandbox pipeline owns the object arena, binds explicit record layouts, copies the original handle aliases on GPU and displays RGB float output. Original device function bodies are unchanged.

The complete 1200 x 800, 10-sample, 488-sphere scene passes the sandbox test on NVIDIA Blackwell. All 960,000 displayed pixels are checked against GPU output. Native comparison over 2,880,000 components has mean absolute error 0.000614175 and 97.593% within 1e-4. Floating-point differences can change stochastic ray paths; this is not exact pixel parity. The standalone run also verifies all object slots are freed. Timings captured during concurrent native and browser runs are diagnostic only, not a speed comparison.
