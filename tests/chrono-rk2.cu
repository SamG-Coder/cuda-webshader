typedef int int32_t;
// Real3/Real4 definitions: Copyright (c) 2025 projectchrono.org.
// Parameter record and grid helpers: Copyright (c) 2014 projectchrono.org.
// BSD terms are reproduced in THIRD_PARTY_NOTICES.md.
// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2024 projectchrono.org
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Author: Radu Serban
// =============================================================================
//
// Miscellaneous enumerations for the SPH-based Chrono::FSI fluid solver
//
// =============================================================================

// Unchanged scoped enums from Chrono a92c6f72f422fbcafe0b37125d4070cb6a3b5803.
enum class PhysicsProblem {
    CFD,  ///< incompressible fluid problem
    CRM   ///< continuous granular problem
};
enum class IntegrationScheme {
    EULER,        ///< Explicit Euler
    RK2,          ///< Runge-Kutta 2
    VERLET,       ///< Velocity Verlet
    SYMPLECTIC,   ///< Symplectic Euler
    IMPLICIT_SPH  ///< Implicit SPH
};
enum class ShiftingMethod { NONE, PPST, XSPH, PPST_XSPH, DIFFUSION, DIFFUSION_XSPH };
enum class EosType { TAIT, ISOTHERMAL };
enum class KernelType { QUADRATIC, CUBIC_SPLINE, QUINTIC_SPLINE, WENDLAND };
enum class ViscosityMethod { LAMINAR, ARTIFICIAL_UNILATERAL, ARTIFICIAL_BILATERAL };
enum class BoundaryMethod { ADAMI, HOLMES };
enum class Rheology { INERTIA_RHEOLOGY, NONLOCAL_FLUIDITY };
enum class FrictionLaw { CONSTANT, LINEAR, NONLINEAR };
enum class SolverType { JACOBI, BICGSTAB, GMRES, CR, CG, SAP };
enum class RheologyCRM { MU_OF_I, MCC };
enum class BCType {
    NONE,         ///< no boundary conditions enforced
    PERIODIC,     ///< periodic boundary conditions
    INLET_OUTLET  ///< inlet-outlet boundary conditions
};
enum class NodeDirections { NONE, AVERAGE, EXACT };
enum class BcePatternMesh1D { FULL, STAR };
enum class BcePatternMesh2D { CENTERED, OUTWARD, INWARD };
enum class OutputLevel {
    STATE,           ///< marker state, velocity, and acceleration
    STATE_PRESSURE,  ///< STATE plus density and pressure
    CFD_FULL,        ///< STATE_PRESSURE plus various CFD parameters
    CRM_FULL         ///< STATE_PRESSURE plus normal and shear stress
};

typedef float Real;
typedef unsigned int uint;
struct Real3 {
    Real x;
    Real y;
    Real z;
};
struct Real4 {
    Real x;
    Real y;
    Real z;
    Real w;
};
struct BoundaryConditions {
    BCType x;
    BCType y;
    BCType z;
};
struct ChFsiParamsSPH {
    PhysicsProblem physics_problem;        ///< Physics problem type: (CFD or CRM)
    RheologyCRM rheology_model_crm;        ///< Rheology model for CRM problems (MU_OF_I or MCC)

    IntegrationScheme integration_scheme;  ///< Integration scheme
    EosType eos_type;                      ///< Equation of state type (Tait or isothermal)
    ViscosityMethod viscosity_method;      ///< Viscosity treatment type (physics-based laminar flow or artificial)
    BoundaryMethod boundary_method;        ///< Boundary type (Adami or Holmes)
    KernelType kernel_type;                ///< Kernel type (Quadratic, cubic spline, quintic spline, quintic Wendland)
    ShiftingMethod shifting_method;        ///< Shifting method (NONE, PPST, XSPH, PPST_XSPH)

    int3 gridSize;          ///< dx, dy, dz distances between particle centers
    Real3 worldOrigin;      ///< Origin point
    Real3 cellSize;         ///< Cell size for the neighbor particle search
    uint numBodies;         ///< Number of FSI bodies.
    Real3 boxDims;          ///< Dimensions (AABB) of the domain
    Real3 zombieBoxDims;    ///< Dimensions (AABB) of the zombie domain
    Real3 zombieOrigin;     ///< Origin point of the zombie domain
    Real d0;                ///< Initial separation of SPH particles
    Real ood0;              ///< 1 / d0
    Real d0_multiplier;     ///< Multiplier to obtain the interaction length, h = d0_multiplier * d0
    Real h;                 ///< Kernel interaction length
    Real ooh;               ///< 1 / h
    Real h_multiplier;      ///< Multiplier to obtain kernel radius, r = h_multiplier * h (depends on kernel type)
    int num_neighbors;      ///< Number of neighbor particles
    Real epsMinMarkersDis;  ///< Multiplier for minimum distance between markers (d_min = eps * h)
    int num_bce_layers;     ///< Number of BCE marker layers attached to boundary and solid surfaces (default: 3)
    Real toleranceZone;     ///< Helps determine the particles that are in the domain but are outside the boundaries, so
                            ///< they are not considered fluid particles and are dropped at the beginning of the simulation.

    Real base_pressure;    ///< Relative value of pressure applied to the whole domain
    Real3 delta_pressure;  ///< Change in Pressure for periodic BC (when particle moves from one side to the other)

    Real3 V_in;  ///< Inlet velocity for inlet BC
    Real x_in;   ///< Inlet position for inlet BC

    Real3 gravity;     ///< Gravitational acceleration
    Real3 bodyForce3;  ///< Constant force applied to the fluid particles (solids not directly affected)

    Real rho0;     ///< Density
    Real invrho0;  ///< 1 / rho0
    Real volume0;  ///< Initial particle volume

    Real markerMass;               ///< marker mass
    Real mu0;                      ///< Viscosity
    Real v_Max;                    ///< Max velocity of fluid used in equation of state. Run simulation once to be able to determine it.
    Real shifting_xsph_eps;        ///< Coefficient for XSPH shifting
    Real shifting_ppst_push;       ///< Coefficient for PPST shifting - this is applied when penetration with fictitious
                                   ///< sphere is detected
    Real shifting_ppst_pull;       ///< Coefficient for PPST pulling - this is applied when penetration with fictitious
    Real shifting_beta_implicit;   ///< Coefficient for shifting used in implicit scheme
    Real shifting_diffusion_A;     ///< Fickian shifting coefficient. Scales the shifting velocity
                                   ///< -A h |v_i| sum_j (m_j / rho_j) grad W_ij, whose sum is the discrete
                                   ///< gradient of particle concentration (default: 1.0, range 1 to 6)
    Real shifting_diffusion_AFSM;  ///< Upper anchor of the free-surface taper applied to diffusion
                                   ///< shifting. Particles whose position-field divergence reaches AFSM are
                                   ///< given the full shift; between AFST and AFSM the shift ramps linearly.
                                   ///< Complete 3D kernel support yields about 2.9 (default: 2.9)
    Real shifting_diffusion_AFST;  ///< Lower anchor of the free-surface taper applied to diffusion
                                   ///< shifting. Particles whose position-field divergence is at or below
                                   ///< AFST are not shifted at all (default: 2.0)

    Real dT;  ///< Time step. Depending on the model this will vary and the only way to determine what time step to
              ///< use is to run simulations multiple time and find which one is the largest dT that produces a
              ///< stable simulation.

    Real kdT;      ///< Implicit integration parameter
    Real gammaBB;  ///< Equation of state parameter

    bool use_default_limits;  ///< true if cMin and cMax are not user-provided (default: true)
    bool use_init_pressure;   ///< true if pressure set based on height (default: false)

    Real3 cMinInit;  ///< Minimum point of the fluid domain.
    Real3 cMaxInit;  ///< Maximum point of the fluid domain.
    Real binSize0;   ///< Suggests the length of the bin each particle occupies. Normally this would be 2*hsml since
                     ///< hsml is the radius of the particle, but when we have periodic boundary condition varies a
                     ///< little from 2 hsml.This may change slightly due to the location of the periodic BC.

    double pressure_height;  ///< height for pressure initialization

    // Note: more frequent re-initialization helps in getting more accurate incompressible fluid,
    // but more stable solution is obtained for larger values of density_reinit_steps
    int density_reinit_steps;  ///< reinitialize density after density_reinit_steps steps

    bool Conservative_Form;  ///< use conservative or consistent discretization
    int gradient_type;       ///< Type of the gradient operator
    int laplacian_type;      ///< Type of the Laplacian operator

    bool use_consistent_gradient_discretization;   ///< use consistent discretization for gradient operator
    bool use_consistent_laplacian_discretization;  ///< use consistent discretization for Laplacian operator
    bool use_delta_sph;                            ///< use delta SPH
    Real density_delta;                            ///< parameter for delta SPH

    bool use_density_based_projection;  ///< Set true to use density based projection scheme in ISPH solver

    bool Pressure_Constraint;  ///< Whether the singularity of the pressure equation should be fixed
    SolverType LinearSolver;   ///< Type of the linear solver

    Real Alpha;  ///< Poisson Pressure Equation source term constant. Used to control the noise in the FS forces

    Real LinearSolver_Abs_Tol;  ///< Poisson Pressure Equation residual
    Real LinearSolver_Rel_Tol;  ///< Poisson Pressure Equation Absolute residual
    int LinearSolver_Max_Iter;  ///< Linear Solver maximum number of iteration
    bool Verbose_monitoring;    ///< Poisson Pressure Equation Absolute residual

    Real Max_Pressure;             ///< Max Pressure in the pressure solver
    Real PPE_relaxation;           ///< PPE_relaxation
    bool ClampPressure;            ///< Clamp pressure to 0 if negative, based on the ISPH paper by Ihmsen et al. (2013)
    Real IncompressibilityFactor;  ///< Incompressibility factor (default: 1)
    Real Cs;                       ///< Speed of sound

    bool Apply_BC_U;        ///< This option lets you apply a velocity BC on the BCE markers
    Real L_Characteristic;  ///< Characteristic for Re number computation

    bool non_newtonian;       ///< Set true to model non-Newtonian fluid
    Rheology rheology_model;  ///< Model of the rheology
    Real ave_diam;            ///< average particle diameter
    Real cohesion;            ///< c in the stress model sigma=(mu*p+c)/|D|
    FrictionLaw mu_of_I;      ///< Constant I in granular material dynamics
    Real mu_max;              ///< maximum viscosity
    Real mu_fric_s;           ///< friction mu_s
    Real mu_fric_2;           ///< mu_2 constant in mu=mu(I)
    Real mu_I0;               ///< Reference Inertia number
    Real mu_I_b;              ///< b constant in mu=mu(I)=mu_s+b*I

    Real HB_sr0;   ///< Herschel–Bulkley consistency index
    Real HB_k;     ///< Herschel–Bulkley consistency index
    Real HB_n;     ///< Herschel–Bulkley  power
    Real HB_tau0;  ///< Herschel–Bulkley yield stress

    Real E_young;                 ///< Young's modulus
    Real G_shear;                 ///< Shear modulus
    Real INV_G_shear;             ///< 1.0 / G_shear
    Real K_bulk;                  ///< Bulk modulus
    Real Nu_poisson;              ///< Poisson's ratio
    Real artificial_viscosity;    ///< Artificial viscosity coefficient
    Real Coh_coeff;               ///< Cohesion coefficient
    Real free_surface_threshold;  ///< threshold for identifying free surface. The divergence of the position
    ///< field is computed and compared to this threshold. Particles with divergence
    ///< less than this threshold are considered free surface particles. Evaluated for
    ///< both CFD and CRM problems, but currently only the CRM solution consumes the
    ///< result (the stress state is zeroed at flagged particles); default: 2.4
    Real mcc_M;         ///< Cam-Clay critical state line slope, q = M p
    Real mcc_kappa;     ///< Cam-Clay swelling index: slope of the elastic unload/reload line in
                        ///< v-ln(p). Sets the elastic bulk modulus, K = v p / kappa.
                        ///< Must satisfy 0 < mcc_kappa < mcc_lambda
    Real mcc_lambda;    ///< Cam-Clay compression index: slope of the normal consolidation line in
                        ///< v-ln(p). Governs virgin compressibility and the hardening rate, which
                        ///< divides by (mcc_lambda - mcc_kappa). Must exceed mcc_kappa
    Real mcc_v_lambda;  ///< Specific volume at reference pressure of 1000 Pa

    Real boxDimX;  ///< Dimension of the space domain - X
    Real boxDimY;  ///< Dimension of the space domain - Y
    Real boxDimZ;  ///< Dimension of the space domain - Z

    BoundaryConditions bc_type;  ///< boundary condition types in the 3 domain directions
    bool x_periodic;             ///< periodic boundary conditions in x direction?
    bool y_periodic;             ///< periodic boundary conditions in y direction?
    bool z_periodic;             ///< periodic boundary conditions in z direction?

    int3 minBounds;  ///< Lower limit point of the grid (in grid index)
    int3 maxBounds;  ///< Upper limit point of the grid (in grid index)

    Real3 cMin;  ///< Lower limit point (in world coordinates)
    Real3 cMax;  ///< Upper limit point (in world coordinates)

    Real3 zombieMin;  ///< Lower limit point of the zombie domain -> All particles outside this will be frozen
    Real3 zombieMax;  ///< Upper limit point of the zombie domain -> All particles outside this will be frozen

    Real free_flow_duration;  ///< initial  duration for free flow CRM material (default: 0)

    int num_proximity_search_steps;  ///< Number of steps between updates to neighbor lists
    bool use_variable_time_step;     ///< use variable time step (default: false)
};
__constant__ static ChFsiParamsSPH paramsD;
__device__ inline int3 calcGridPos(Real3 p) {
    int3 gridPos;

    gridPos.x = (int)floor((p.x - paramsD.worldOrigin.x) / (paramsD.cellSize.x));
    gridPos.y = (int)floor((p.y - paramsD.worldOrigin.y) / (paramsD.cellSize.y));
    gridPos.z = (int)floor((p.z - paramsD.worldOrigin.z) / (paramsD.cellSize.z));
    return gridPos;
}
__device__ inline int reduceGridIndex(int i, int n, bool periodic) {
    if (periodic) {
        i %= n;
        return (i < 0) ? i + n : i;
    }
    return (i < 0) ? 0 : (i >= n) ? n - 1 : i;
}
__device__ inline uint calcGridHash(int3 gridPos) {
    gridPos.x = reduceGridIndex(gridPos.x, paramsD.gridSize.x, paramsD.x_periodic);
    gridPos.y = reduceGridIndex(gridPos.y, paramsD.gridSize.y, paramsD.y_periodic);
    gridPos.z = reduceGridIndex(gridPos.z, paramsD.gridSize.z, paramsD.z_periodic);

    return gridPos.z * paramsD.gridSize.y * paramsD.gridSize.x + gridPos.y * paramsD.gridSize.x + gridPos.x;
}

#define mR3 make_Real3
#define mI3 make_int3
#define CH_FSI_SPH_NOINLINE __declspec(noinline)
__host__ __device__ inline Real3 make_Real3(Real a, Real b, Real c)  ///
{
    Real3 d;
    d.x = a;
    d.y = b;
    d.z = c;
    return d;
}
__host__ __device__ inline Real3 make_Real3(Real4 a) {
    return make_Real3(a.x, a.y, a.z);
}
__host__ __device__ inline Real3 operator-(Real3 a, Real3 b) {
    return make_Real3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ inline CH_FSI_SPH_NOINLINE Real MinimumImageShiftMultiPeriod(Real dist, Real period) {
    return period * rint(dist / period);
}
__device__ inline Real MinimumImageShift(Real dist, Real period) {
    // A non-periodic axis exits first and must: it has period == 0, and falling through to the
    // division would evaluate 0 * rint(dist / 0), which is 0 * inf, which is NaN, in the distance
    // function every force kernel uses.
    if (period <= 0)
        return Real(0);
    // The common case costs two comparisons and no call, which is what it cost before the fix.
    Real half = Real(0.5) * period;
    if (dist <= half && dist >= -half)
        return Real(0);
    return MinimumImageShiftMultiPeriod(dist, period);
}
__device__ inline Real3 Modify_Local_PosB(Real3& b, Real3 a) {
    Real3 dist3 = a - b;
    b.x += MinimumImageShift(dist3.x, paramsD.x_periodic ? paramsD.boxDims.x : Real(0));
    b.y += MinimumImageShift(dist3.y, paramsD.y_periodic ? paramsD.boxDims.y : Real(0));
    b.z += MinimumImageShift(dist3.z, paramsD.z_periodic ? paramsD.boxDims.z : Real(0));

    dist3 = a - b;
    // modifying the markers perfect overlap
    Real dd = dist3.x * dist3.x + dist3.y * dist3.y + dist3.z * dist3.z;
    Real MinD = paramsD.epsMinMarkersDis * paramsD.h;
    Real sq_MinD = MinD * MinD;
    if (dd < sq_MinD) {
        dist3 = mR3(MinD, 0, 0);
    }
    b = a - dist3;
    return (dist3);
}
__device__ inline Real3 Distance(Real3 a, Real3 b) {
    return Modify_Local_PosB(b, a);
}
inline __host__ __device__ int3 operator+(int3 a, int3 b) {
    return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__global__ void neighborSearchNum(const Real4* sortedPosRad,
                                  const Real4* sortedRhoPreMu,
                                  const uint* cellStart,
                                  const uint* cellEnd,
                                  const uint numActive,
                                  uint* numNeighborsPerPart) {
    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numActive) {
        return;
    }

    Real3 posRadA = mR3(sortedPosRad[index]);
    int3 gridPos = calcGridPos(posRadA);
    Real SuppRadii = 2.0f * paramsD.h;
    Real SqRadii = SuppRadii * SuppRadii;
    uint j_num = 0;

    for (int z = -1; z <= 1; z++) {
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                int3 neighborPos = gridPos + mI3(x, y, z);
                // Check if we need to skip this neighbor position (out of bounds for non-periodic dimensions)
                if (neighborPos.x < paramsD.minBounds.x || neighborPos.x > paramsD.maxBounds.x || neighborPos.y < paramsD.minBounds.y || neighborPos.y > paramsD.maxBounds.y ||
                    neighborPos.z < paramsD.minBounds.z || neighborPos.z > paramsD.maxBounds.z) {
                    continue;
                }
                uint gridHash = calcGridHash(neighborPos);
                uint startIndex = cellStart[gridHash];
                uint endIndex = cellEnd[gridHash];
                for (uint j = startIndex; j < endIndex; j++) {
                    Real3 posRadB = mR3(sortedPosRad[j]);
                    Real3 dist3 = Distance(posRadA, posRadB);
                    Real dd = dist3.x * dist3.x + dist3.y * dist3.y + dist3.z * dist3.z;
                    if (dd < SqRadii) {
                        j_num++;
                    }
                }
            }
        }
    }
    numNeighborsPerPart[index] = j_num;
}
__global__ void neighborSearchID(const Real4* sortedPosRad,
                                 const Real4* sortedRhoPreMu,
                                 const uint* cellStart,
                                 const uint* cellEnd,
                                 const uint numActive,
                                 const uint* numNeighborsPerPart,
                                 uint* neighborList) {
    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numActive) {
        return;
    }
    Real3 posRadA = mR3(sortedPosRad[index]);
    int3 gridPos = calcGridPos(posRadA);
    Real SuppRadii = 2.0f * paramsD.h;
    Real SqRadii = SuppRadii * SuppRadii;
    uint j_num = 1;
    neighborList[numNeighborsPerPart[index]] = index;

    for (int z = -1; z <= 1; z++) {
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                int3 neighborPos = gridPos + mI3(x, y, z);
                // Check if we need to skip this neighbor position (out of bounds for non-periodic dimensions)
                if (neighborPos.x < paramsD.minBounds.x || neighborPos.x > paramsD.maxBounds.x || neighborPos.y < paramsD.minBounds.y || neighborPos.y > paramsD.maxBounds.y ||
                    neighborPos.z < paramsD.minBounds.z || neighborPos.z > paramsD.maxBounds.z) {
                    continue;
                }
                uint gridHash = calcGridHash(neighborPos);
                uint startIndex = cellStart[gridHash];
                uint endIndex = cellEnd[gridHash];
                for (uint j = startIndex; j < endIndex; j++) {
                    if (j != index) {
                        Real3 posRadB = mR3(sortedPosRad[j]);
                        Real3 dist3 = Distance(posRadA, posRadB);
                        Real dd = dist3.x * dist3.x + dist3.y * dist3.y + dist3.z * dist3.z;
                        if (dd < SqRadii) {
                            neighborList[numNeighborsPerPart[index] + j_num] = j;
                            j_num++;
                        }
                    }
                }
            }
        }
    }
}


#define UINT_MAX 4294967295u
__global__ void findCellStartEndD(uint* cellStartD,        // output: cell start index
                                  uint* cellEndD,          // output: cell end index
                                  uint* gridMarkerHashD,   // input: sorted grid hashes
                                  uint* gridMarkerIndexD,  // input: sorted particle indices
                                  uint numActive) {
    extern __shared__ uint sharedHash[];  // blockSize + 1 elements
    // Get the particle index the current thread is supposed to be looking at.
    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    uint hash;

    if (index >= numActive)
        return;

    hash = gridMarkerHashD[index];
    if (hash == UINT_MAX)
        return;
    // Load hash data into shared memory so that we can look at neighboring
    // particle's hash value without loading two hash values per thread
    sharedHash[threadIdx.x + 1] = hash;

    // first thread in block must load neighbor particle hash
    if (index > 0 && threadIdx.x == 0)
        sharedHash[0] = gridMarkerHashD[index - 1];

    __syncthreads();
    if (sharedHash[threadIdx.x] == UINT_MAX)
        return;

    // If this particle has a different cell index to the previous
    // particle then it must be the first particle in the cell,
    // so store the index of this particle in the cell. As it
    // isn't the first particle, it must also be the cell end of
    // the previous particle's cell.
    if (index == 0 || hash != sharedHash[threadIdx.x]) {
        cellStartD[hash] = index;
        if (index > 0)
            cellEndD[sharedHash[threadIdx.x]] = index;
    }

    if (index == numActive - 1)
        cellEndD[hash] = index + 1;
}

// MIT data preparation wrappers; distance and search code above is unchanged.
__global__ void hashMarkers(const Real4* positions,uint* hashes,uint* indices,uint n){uint i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){hashes[i]=calcGridHash(calcGridPos(mR3(positions[i])));indices[i]=i;}}
__global__ void gatherMarkers(const Real4* positions,const uint* indices,Real4* sortedPosRad,uint n){uint i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)sortedPosRad[i]=positions[indices[i]];}
// MIT selection adapter: preserve original marker IDs through the existing grid helpers.
__global__ void hashSelected(const Real4* positions,const uint* activeList,uint* hashes,uint* indices,uint n){uint i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){uint original=activeList[i];hashes[i]=calcGridHash(calcGridPos(mR3(positions[original])));indices[i]=original;}}

#define INVPI Real(0.31830988618379)
#define EPSILON Real(1e-8)
inline __host__ __device__ Real square(Real a) {
    return a * a;
}
inline __host__ __device__ Real cube(Real a) {
    return a * a * a;
}
inline __host__ __device__ Real quartic(Real a) {
    return a * a * a * a;
}
inline __host__ __device__ Real quintic(Real a) {
    return a * a * a * a * a;
}
inline __host__ __device__ Real W3h_CubicSpline(Real d, Real invh) {
    Real q = fabs(d) * invh;

    if (q < 1) {
        Real alpha = INVPI * cube(invh) / 4;
        return alpha * (cube(2 - q) - 4 * cube(1 - q));
    }
    if (q < 2) {
        Real alpha = INVPI * cube(invh) / 4;
        return alpha * cube(2 - q);
    }
    return 0;
}
inline __host__ __device__ Real W3h_Quadratic(Real d, Real invh) {
    Real q = fabs(d) * invh;
    if (q < 2) {
        Real alpha = (15 * INVPI * cube(invh)) / 16;
        return alpha * (square(q) / 4 - q + 1);
    }
    return 0;
}
inline __host__ __device__ Real W3h_QuinticSpline(Real d, Real invh) {
    Real q = fabs(d) * invh;
    Real alpha = INVPI * cube(invh) / 120;

    if (q < 1) {
        return alpha * (quintic(3 - q) - 6 * quintic(2 - q) + 15 * quintic(1 - q));
    }
    if (q < 2) {
        return alpha * (quintic(3 - q) - 6 * quintic(2 - q));
    }
    if (q < 3) {
        return alpha * (quintic(3 - q));
    }
    return 0;
}
inline __host__ __device__ Real W3h_Wendland(Real d, Real invh) {
    Real q = fabs(d) * invh;

    if (q < 2) {
        Real alpha = 21 * INVPI * cube(invh) / 256;
        return alpha * quartic(2 - q) * (2 * q + 1);
    }
    return 0;
}
inline __host__ __device__ Real W3h(KernelType type, Real d, Real invh) {
    switch (type) {
        case KernelType::QUADRATIC:
            return W3h_Quadratic(d, invh);
        case KernelType::CUBIC_SPLINE:
            return W3h_CubicSpline(d, invh);
        case KernelType::QUINTIC_SPLINE:
            return W3h_QuinticSpline(d, invh);
        case KernelType::WENDLAND:
            return W3h_Wendland(d, invh);
    }

    return -1;
}
inline __device__ Real InvEos(Real pw, EosType eos_type) {
    switch (eos_type) {
        case EosType::TAIT: {
            Real gama = 7;
            Real B = paramsD.rho0 * paramsD.Cs * paramsD.Cs / gama;
            Real powerComp = (pw - paramsD.base_pressure) / B + 1.0;
            Real rho = (powerComp > 0) ? paramsD.rho0 * pow(powerComp, 1.0 / gama) : -paramsD.rho0 * pow(fabs(powerComp), 1.0 / gama);
            return rho;
        }

        case EosType::ISOTHERMAL: {
            Real rho = pw / (paramsD.Cs * paramsD.Cs) + paramsD.rho0;
            return rho;
        }
    }
    return -1;
}
__host__ __device__ inline bool IsFluidParticle(Real code) {
    return code < -0.5;
}
__host__ __device__ inline bool IsBceMarker(Real code) {
    return code > -0.5;
}
__host__ __device__ inline bool IsBceSolidMarker(Real code) {
    return code > 0.5;
}


__host__ __device__ inline Real3 make_Real3(Real s) {
    return make_Real3(s, s, s);
}
__host__ __device__ inline Real3 operator+(Real3 a, Real3 b) {
    return make_Real3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__host__ __device__ inline void operator+=(Real3& a, Real3 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

__host__ __device__ inline Real3 operator*(Real3 a, Real b) {
    return make_Real3(a.x * b, a.y * b, a.z * b);
}
__host__ __device__ inline Real3 operator*(Real b, Real3 a) {
    return make_Real3(b * a.x, b * a.y, b * a.z);
}
__host__ __device__ inline Real3 operator/(Real3 a, Real b) {
    return make_Real3(a.x / b, a.y / b, a.z / b);
}
__host__ __device__ inline Real dot(Real3 a, Real3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__host__ __device__ inline Real length(Real3 v) {
    return sqrt(dot(v, v));
}

__global__ void CfdAdamiBC_D(const uint* numNeighborsPerPart,
                             const uint* neighborList,
                             const Real4* sortedPosRadD,
                             const uint numActive,
                             Real3* bceAcc,
                             Real4* sortedRhoPresMuD,
                             Real3* sortedVelMasD,
                             volatile bool* error_flag) {
    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numActive)
        return;

    // Ignore all fluid particles
    if (IsFluidParticle(sortedRhoPresMuD[index].w)) {
        return;
    }

    Real3 posRadA = mR3(sortedPosRadD[index]);
    uint NLStart = numNeighborsPerPart[index];
    uint NLEnd = numNeighborsPerPart[index + 1];
    Real sum_pw = 0;
    Real3 sum_rhorw = mR3(0);
    Real sum_w = 0;
    Real3 sum_vw = mR3(0);

    for (int n = NLStart + 1; n < NLEnd; n++) {
        uint j = neighborList[n];

        // only consider fluid neighbors
        if (IsBceMarker(sortedRhoPresMuD[j].w)) {
            continue;
        }

        Real3 posRadB = mR3(sortedPosRadD[j]);
        Real3 rij = Distance(posRadA, posRadB);
        Real d = length(rij);
        Real W3 = W3h(paramsD.kernel_type, d, paramsD.ooh);
        sum_w += W3;
        sum_pw += sortedRhoPresMuD[j].y * W3;
        sum_rhorw += sortedRhoPresMuD[j].x * rij * W3;
        sum_vw += sortedVelMasD[j] * W3;
    }

    if (sum_w > EPSILON) {
        Real3 prescribedVel = (IsBceSolidMarker(sortedRhoPresMuD[index].w)) ? (2.0f * sortedVelMasD[index]) : mR3(0);
        sortedVelMasD[index] = prescribedVel - sum_vw / sum_w;
        sortedRhoPresMuD[index].y = (sum_pw + dot(paramsD.gravity - bceAcc[index], sum_rhorw)) / sum_w;
        sortedRhoPresMuD[index].x = InvEos(sortedRhoPresMuD[index].y, paramsD.eos_type);
    } else {
        sortedVelMasD[index] = mR3(0);
        sortedRhoPresMuD[index].y = 0;
        sortedVelMasD[index] = mR3(0);
    }
}

#define mR4 make_Real4
__host__ __device__ inline Real4 make_Real4(Real a, Real b, Real c, Real d)  ///
{
    Real4 e;
    e.x = a;
    e.y = b;
    e.z = c;
    e.w = d;
    return e;
}
__host__ __device__ inline Real4 make_Real4(Real s) {
    return make_Real4(s, s, s, s);
}
__host__ __device__ inline Real4 make_Real4(Real3 a) {
    return make_Real4(a.x, a.y, a.z, 0.0);
}
__host__ __device__ inline Real4 make_Real4(Real3 a, Real w) {
    return make_Real4(a.x, a.y, a.z, w);
}
__host__ __device__ inline Real3 operator-(Real3& a) {
    return make_Real3(-a.x, -a.y, -a.z);
}
__host__ __device__ inline void operator+=(Real4& a, Real4 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
__host__ __device__ inline bool IsFinite(Real4 v) {
#ifdef __CUDA_ARCH__
    return isfinite(v.x) && isfinite(v.y) && isfinite(v.z) && isfinite(v.w);
#else
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z) && std::isfinite(v.w);
#endif
}
inline __host__ __device__ Real3 GradW3h_CubicSpline(Real3 d, Real invh) {
    Real q = length(d) * invh;
    if (abs(q) < EPSILON)
        return mR3(0);

    // beta = 3 * alpha / h^2
    if (q < 1) {
        Real beta = 3 * INVPI * quintic(invh) / 4;
        return (beta * (3 * q - 4)) * d;
    }
    if (q < 2) {
        Real beta = 3 * INVPI * quintic(invh) / 4;
        return (beta * (4 - q - 4 / q)) * d;
    }
    return mR3(0);
}
inline __host__ __device__ Real3 GradW3h_Quadratic(Real3 d, Real invh) {
    Real q = length(d) * invh;
    if (abs(q) < EPSILON)
        return mR3(0);

    if (q < 2) {
        // beta = 1/2 * alpha / h^2
        Real beta = (15 * INVPI * quintic(invh)) / 32;
        return (beta * (1 - 2 / q)) * d;
    }
    return mR3(0);
}
inline __host__ __device__ Real3 GradW3h_QuinticSpline(Real3 d, Real invh) {
    Real q = length(d) * invh;
    if (fabs(q) < 1e-10)
        return mR3(0);

    // beta = -5 * alpha / h^2
    Real beta = -5 * INVPI * quintic(invh) / 120;
    if (q < 1) {
        return ((beta / q) * (quartic(3 - q) - 6 * quartic(2 - q) + 15 * quartic(1 - q))) * d;
    }
    if (q < 2) {
        return ((beta / q) * (quartic(3 - q) - 6 * quartic(2 - q))) * d;
    }
    if (q < 3) {
        return ((beta / q) * (quartic(3 - q))) * d;
    }
    return mR3(0);
}
inline __host__ __device__ Real3 GradW3h_Wendland(Real3 d, Real invh) {
    Real q = length(d) * invh;
    if (fabs(q) < 1e-10)
        return mR3(0);

    if (q < 2) {
        // beta = -10 * alpha / h^2
        Real beta = -210 * INVPI * quintic(invh) / 256;
        return (beta * cube(2 - q)) * d;
    }
    return mR3(0);
}
inline __host__ __device__ Real3 GradW3h(KernelType type, Real3 d, Real invh) {
    switch (type) {
        case KernelType::QUADRATIC:
            return GradW3h_Quadratic(d, invh);
        case KernelType::CUBIC_SPLINE:
            return GradW3h_CubicSpline(d, invh);
        case KernelType::QUINTIC_SPLINE:
            return GradW3h_QuinticSpline(d, invh);
        case KernelType::WENDLAND:
            return GradW3h_Wendland(d, invh);
    }

    return mR3(-1, -1, -1);
}
__host__ __device__ inline bool IsBceWallMarker(Real code) {
    return code > -0.5 && code < 0.5;
}
__host__ __device__ inline bool IsSphParticle(Real code) {
    return code < -0.5 && code > -1.5;
}
static __device__ void inv6xdelta_mn(Real* B, Real* L) {
    Real DET = B[0] * B[7] * B[14] * B[21] * B[28] * B[35] - B[0] * B[7] * B[14] * B[21] * B[29] * B[34] - B[0] * B[7] * B[14] * B[22] * B[27] * B[35] +
               B[0] * B[7] * B[14] * B[22] * B[29] * B[33] + B[0] * B[7] * B[14] * B[23] * B[27] * B[34] - B[0] * B[7] * B[14] * B[23] * B[28] * B[33] -
               B[0] * B[7] * B[15] * B[20] * B[28] * B[35] + B[0] * B[7] * B[15] * B[20] * B[29] * B[34] + B[0] * B[7] * B[15] * B[22] * B[26] * B[35] -
               B[0] * B[7] * B[15] * B[22] * B[29] * B[32] - B[0] * B[7] * B[15] * B[23] * B[26] * B[34] + B[0] * B[7] * B[15] * B[23] * B[28] * B[32] +
               B[0] * B[7] * B[16] * B[20] * B[27] * B[35] - B[0] * B[7] * B[16] * B[20] * B[29] * B[33] - B[0] * B[7] * B[16] * B[21] * B[26] * B[35] +
               B[0] * B[7] * B[16] * B[21] * B[29] * B[32] + B[0] * B[7] * B[16] * B[23] * B[26] * B[33] - B[0] * B[7] * B[16] * B[23] * B[27] * B[32] -
               B[0] * B[7] * B[17] * B[20] * B[27] * B[34] + B[0] * B[7] * B[17] * B[20] * B[28] * B[33] + B[0] * B[7] * B[17] * B[21] * B[26] * B[34] -
               B[0] * B[7] * B[17] * B[21] * B[28] * B[32] - B[0] * B[7] * B[17] * B[22] * B[26] * B[33] + B[0] * B[7] * B[17] * B[22] * B[27] * B[32] -
               B[0] * B[8] * B[13] * B[21] * B[28] * B[35] + B[0] * B[8] * B[13] * B[21] * B[29] * B[34] + B[0] * B[8] * B[13] * B[22] * B[27] * B[35] -
               B[0] * B[8] * B[13] * B[22] * B[29] * B[33] - B[0] * B[8] * B[13] * B[23] * B[27] * B[34] + B[0] * B[8] * B[13] * B[23] * B[28] * B[33] +
               B[0] * B[8] * B[15] * B[19] * B[28] * B[35] - B[0] * B[8] * B[15] * B[19] * B[29] * B[34] - B[0] * B[8] * B[15] * B[22] * B[25] * B[35] +
               B[0] * B[8] * B[15] * B[22] * B[29] * B[31] + B[0] * B[8] * B[15] * B[23] * B[25] * B[34] - B[0] * B[8] * B[15] * B[23] * B[28] * B[31] -
               B[0] * B[8] * B[16] * B[19] * B[27] * B[35] + B[0] * B[8] * B[16] * B[19] * B[29] * B[33] + B[0] * B[8] * B[16] * B[21] * B[25] * B[35] -
               B[0] * B[8] * B[16] * B[21] * B[29] * B[31] - B[0] * B[8] * B[16] * B[23] * B[25] * B[33] + B[0] * B[8] * B[16] * B[23] * B[27] * B[31] +
               B[0] * B[8] * B[17] * B[19] * B[27] * B[34] - B[0] * B[8] * B[17] * B[19] * B[28] * B[33] - B[0] * B[8] * B[17] * B[21] * B[25] * B[34] +
               B[0] * B[8] * B[17] * B[21] * B[28] * B[31] + B[0] * B[8] * B[17] * B[22] * B[25] * B[33] - B[0] * B[8] * B[17] * B[22] * B[27] * B[31] +
               B[0] * B[9] * B[13] * B[20] * B[28] * B[35] - B[0] * B[9] * B[13] * B[20] * B[29] * B[34] - B[0] * B[9] * B[13] * B[22] * B[26] * B[35] +
               B[0] * B[9] * B[13] * B[22] * B[29] * B[32] + B[0] * B[9] * B[13] * B[23] * B[26] * B[34] - B[0] * B[9] * B[13] * B[23] * B[28] * B[32] -
               B[0] * B[9] * B[14] * B[19] * B[28] * B[35] + B[0] * B[9] * B[14] * B[19] * B[29] * B[34] + B[0] * B[9] * B[14] * B[22] * B[25] * B[35] -
               B[0] * B[9] * B[14] * B[22] * B[29] * B[31] - B[0] * B[9] * B[14] * B[23] * B[25] * B[34] + B[0] * B[9] * B[14] * B[23] * B[28] * B[31] +
               B[0] * B[9] * B[16] * B[19] * B[26] * B[35] - B[0] * B[9] * B[16] * B[19] * B[29] * B[32] - B[0] * B[9] * B[16] * B[20] * B[25] * B[35] +
               B[0] * B[9] * B[16] * B[20] * B[29] * B[31] + B[0] * B[9] * B[16] * B[23] * B[25] * B[32] - B[0] * B[9] * B[16] * B[23] * B[26] * B[31] -
               B[0] * B[9] * B[17] * B[19] * B[26] * B[34] + B[0] * B[9] * B[17] * B[19] * B[28] * B[32] + B[0] * B[9] * B[17] * B[20] * B[25] * B[34] -
               B[0] * B[9] * B[17] * B[20] * B[28] * B[31] - B[0] * B[9] * B[17] * B[22] * B[25] * B[32] + B[0] * B[9] * B[17] * B[22] * B[26] * B[31] -
               B[0] * B[10] * B[13] * B[20] * B[27] * B[35] + B[0] * B[10] * B[13] * B[20] * B[29] * B[33] + B[0] * B[10] * B[13] * B[21] * B[26] * B[35] -
               B[0] * B[10] * B[13] * B[21] * B[29] * B[32] - B[0] * B[10] * B[13] * B[23] * B[26] * B[33] + B[0] * B[10] * B[13] * B[23] * B[27] * B[32] +
               B[0] * B[10] * B[14] * B[19] * B[27] * B[35] - B[0] * B[10] * B[14] * B[19] * B[29] * B[33] - B[0] * B[10] * B[14] * B[21] * B[25] * B[35] +
               B[0] * B[10] * B[14] * B[21] * B[29] * B[31] + B[0] * B[10] * B[14] * B[23] * B[25] * B[33] - B[0] * B[10] * B[14] * B[23] * B[27] * B[31] -
               B[0] * B[10] * B[15] * B[19] * B[26] * B[35] + B[0] * B[10] * B[15] * B[19] * B[29] * B[32] + B[0] * B[10] * B[15] * B[20] * B[25] * B[35] -
               B[0] * B[10] * B[15] * B[20] * B[29] * B[31] - B[0] * B[10] * B[15] * B[23] * B[25] * B[32] + B[0] * B[10] * B[15] * B[23] * B[26] * B[31] +
               B[0] * B[10] * B[17] * B[19] * B[26] * B[33] - B[0] * B[10] * B[17] * B[19] * B[27] * B[32] - B[0] * B[10] * B[17] * B[20] * B[25] * B[33] +
               B[0] * B[10] * B[17] * B[20] * B[27] * B[31] + B[0] * B[10] * B[17] * B[21] * B[25] * B[32] - B[0] * B[10] * B[17] * B[21] * B[26] * B[31] +
               B[0] * B[11] * B[13] * B[20] * B[27] * B[34] - B[0] * B[11] * B[13] * B[20] * B[28] * B[33] - B[0] * B[11] * B[13] * B[21] * B[26] * B[34] +
               B[0] * B[11] * B[13] * B[21] * B[28] * B[32] + B[0] * B[11] * B[13] * B[22] * B[26] * B[33] - B[0] * B[11] * B[13] * B[22] * B[27] * B[32] -
               B[0] * B[11] * B[14] * B[19] * B[27] * B[34] + B[0] * B[11] * B[14] * B[19] * B[28] * B[33] + B[0] * B[11] * B[14] * B[21] * B[25] * B[34] -
               B[0] * B[11] * B[14] * B[21] * B[28] * B[31] - B[0] * B[11] * B[14] * B[22] * B[25] * B[33] + B[0] * B[11] * B[14] * B[22] * B[27] * B[31] +
               B[0] * B[11] * B[15] * B[19] * B[26] * B[34] - B[0] * B[11] * B[15] * B[19] * B[28] * B[32] - B[0] * B[11] * B[15] * B[20] * B[25] * B[34] +
               B[0] * B[11] * B[15] * B[20] * B[28] * B[31] + B[0] * B[11] * B[15] * B[22] * B[25] * B[32] - B[0] * B[11] * B[15] * B[22] * B[26] * B[31] -
               B[0] * B[11] * B[16] * B[19] * B[26] * B[33] + B[0] * B[11] * B[16] * B[19] * B[27] * B[32] + B[0] * B[11] * B[16] * B[20] * B[25] * B[33] -
               B[0] * B[11] * B[16] * B[20] * B[27] * B[31] - B[0] * B[11] * B[16] * B[21] * B[25] * B[32] + B[0] * B[11] * B[16] * B[21] * B[26] * B[31] -
               B[1] * B[6] * B[14] * B[21] * B[28] * B[35] + B[1] * B[6] * B[14] * B[21] * B[29] * B[34] + B[1] * B[6] * B[14] * B[22] * B[27] * B[35] -
               B[1] * B[6] * B[14] * B[22] * B[29] * B[33] - B[1] * B[6] * B[14] * B[23] * B[27] * B[34] + B[1] * B[6] * B[14] * B[23] * B[28] * B[33] +
               B[1] * B[6] * B[15] * B[20] * B[28] * B[35] - B[1] * B[6] * B[15] * B[20] * B[29] * B[34] - B[1] * B[6] * B[15] * B[22] * B[26] * B[35] +
               B[1] * B[6] * B[15] * B[22] * B[29] * B[32] + B[1] * B[6] * B[15] * B[23] * B[26] * B[34] - B[1] * B[6] * B[15] * B[23] * B[28] * B[32] -
               B[1] * B[6] * B[16] * B[20] * B[27] * B[35] + B[1] * B[6] * B[16] * B[20] * B[29] * B[33] + B[1] * B[6] * B[16] * B[21] * B[26] * B[35] -
               B[1] * B[6] * B[16] * B[21] * B[29] * B[32] - B[1] * B[6] * B[16] * B[23] * B[26] * B[33] + B[1] * B[6] * B[16] * B[23] * B[27] * B[32] +
               B[1] * B[6] * B[17] * B[20] * B[27] * B[34] - B[1] * B[6] * B[17] * B[20] * B[28] * B[33] - B[1] * B[6] * B[17] * B[21] * B[26] * B[34] +
               B[1] * B[6] * B[17] * B[21] * B[28] * B[32] + B[1] * B[6] * B[17] * B[22] * B[26] * B[33] - B[1] * B[6] * B[17] * B[22] * B[27] * B[32] +
               B[1] * B[8] * B[12] * B[21] * B[28] * B[35] - B[1] * B[8] * B[12] * B[21] * B[29] * B[34] - B[1] * B[8] * B[12] * B[22] * B[27] * B[35] +
               B[1] * B[8] * B[12] * B[22] * B[29] * B[33] + B[1] * B[8] * B[12] * B[23] * B[27] * B[34] - B[1] * B[8] * B[12] * B[23] * B[28] * B[33] -
               B[1] * B[8] * B[15] * B[18] * B[28] * B[35] + B[1] * B[8] * B[15] * B[18] * B[29] * B[34] + B[1] * B[8] * B[15] * B[22] * B[24] * B[35] -
               B[1] * B[8] * B[15] * B[22] * B[29] * B[30] - B[1] * B[8] * B[15] * B[23] * B[24] * B[34] + B[1] * B[8] * B[15] * B[23] * B[28] * B[30] +
               B[1] * B[8] * B[16] * B[18] * B[27] * B[35] - B[1] * B[8] * B[16] * B[18] * B[29] * B[33] - B[1] * B[8] * B[16] * B[21] * B[24] * B[35] +
               B[1] * B[8] * B[16] * B[21] * B[29] * B[30] + B[1] * B[8] * B[16] * B[23] * B[24] * B[33] - B[1] * B[8] * B[16] * B[23] * B[27] * B[30] -
               B[1] * B[8] * B[17] * B[18] * B[27] * B[34] + B[1] * B[8] * B[17] * B[18] * B[28] * B[33] + B[1] * B[8] * B[17] * B[21] * B[24] * B[34] -
               B[1] * B[8] * B[17] * B[21] * B[28] * B[30] - B[1] * B[8] * B[17] * B[22] * B[24] * B[33] + B[1] * B[8] * B[17] * B[22] * B[27] * B[30] -
               B[1] * B[9] * B[12] * B[20] * B[28] * B[35] + B[1] * B[9] * B[12] * B[20] * B[29] * B[34] + B[1] * B[9] * B[12] * B[22] * B[26] * B[35] -
               B[1] * B[9] * B[12] * B[22] * B[29] * B[32] - B[1] * B[9] * B[12] * B[23] * B[26] * B[34] + B[1] * B[9] * B[12] * B[23] * B[28] * B[32] +
               B[1] * B[9] * B[14] * B[18] * B[28] * B[35] - B[1] * B[9] * B[14] * B[18] * B[29] * B[34] - B[1] * B[9] * B[14] * B[22] * B[24] * B[35] +
               B[1] * B[9] * B[14] * B[22] * B[29] * B[30] + B[1] * B[9] * B[14] * B[23] * B[24] * B[34] - B[1] * B[9] * B[14] * B[23] * B[28] * B[30] -
               B[1] * B[9] * B[16] * B[18] * B[26] * B[35] + B[1] * B[9] * B[16] * B[18] * B[29] * B[32] + B[1] * B[9] * B[16] * B[20] * B[24] * B[35] -
               B[1] * B[9] * B[16] * B[20] * B[29] * B[30] - B[1] * B[9] * B[16] * B[23] * B[24] * B[32] + B[1] * B[9] * B[16] * B[23] * B[26] * B[30] +
               B[1] * B[9] * B[17] * B[18] * B[26] * B[34] - B[1] * B[9] * B[17] * B[18] * B[28] * B[32] - B[1] * B[9] * B[17] * B[20] * B[24] * B[34] +
               B[1] * B[9] * B[17] * B[20] * B[28] * B[30] + B[1] * B[9] * B[17] * B[22] * B[24] * B[32] - B[1] * B[9] * B[17] * B[22] * B[26] * B[30] +
               B[1] * B[10] * B[12] * B[20] * B[27] * B[35] - B[1] * B[10] * B[12] * B[20] * B[29] * B[33] - B[1] * B[10] * B[12] * B[21] * B[26] * B[35] +
               B[1] * B[10] * B[12] * B[21] * B[29] * B[32] + B[1] * B[10] * B[12] * B[23] * B[26] * B[33] - B[1] * B[10] * B[12] * B[23] * B[27] * B[32] -
               B[1] * B[10] * B[14] * B[18] * B[27] * B[35] + B[1] * B[10] * B[14] * B[18] * B[29] * B[33] + B[1] * B[10] * B[14] * B[21] * B[24] * B[35] -
               B[1] * B[10] * B[14] * B[21] * B[29] * B[30] - B[1] * B[10] * B[14] * B[23] * B[24] * B[33] + B[1] * B[10] * B[14] * B[23] * B[27] * B[30] +
               B[1] * B[10] * B[15] * B[18] * B[26] * B[35] - B[1] * B[10] * B[15] * B[18] * B[29] * B[32] - B[1] * B[10] * B[15] * B[20] * B[24] * B[35] +
               B[1] * B[10] * B[15] * B[20] * B[29] * B[30] + B[1] * B[10] * B[15] * B[23] * B[24] * B[32] - B[1] * B[10] * B[15] * B[23] * B[26] * B[30] -
               B[1] * B[10] * B[17] * B[18] * B[26] * B[33] + B[1] * B[10] * B[17] * B[18] * B[27] * B[32] + B[1] * B[10] * B[17] * B[20] * B[24] * B[33] -
               B[1] * B[10] * B[17] * B[20] * B[27] * B[30] - B[1] * B[10] * B[17] * B[21] * B[24] * B[32] + B[1] * B[10] * B[17] * B[21] * B[26] * B[30] -
               B[1] * B[11] * B[12] * B[20] * B[27] * B[34] + B[1] * B[11] * B[12] * B[20] * B[28] * B[33] + B[1] * B[11] * B[12] * B[21] * B[26] * B[34] -
               B[1] * B[11] * B[12] * B[21] * B[28] * B[32] - B[1] * B[11] * B[12] * B[22] * B[26] * B[33] + B[1] * B[11] * B[12] * B[22] * B[27] * B[32] +
               B[1] * B[11] * B[14] * B[18] * B[27] * B[34] - B[1] * B[11] * B[14] * B[18] * B[28] * B[33] - B[1] * B[11] * B[14] * B[21] * B[24] * B[34] +
               B[1] * B[11] * B[14] * B[21] * B[28] * B[30] + B[1] * B[11] * B[14] * B[22] * B[24] * B[33] - B[1] * B[11] * B[14] * B[22] * B[27] * B[30] -
               B[1] * B[11] * B[15] * B[18] * B[26] * B[34] + B[1] * B[11] * B[15] * B[18] * B[28] * B[32] + B[1] * B[11] * B[15] * B[20] * B[24] * B[34] -
               B[1] * B[11] * B[15] * B[20] * B[28] * B[30] - B[1] * B[11] * B[15] * B[22] * B[24] * B[32] + B[1] * B[11] * B[15] * B[22] * B[26] * B[30] +
               B[1] * B[11] * B[16] * B[18] * B[26] * B[33] - B[1] * B[11] * B[16] * B[18] * B[27] * B[32] - B[1] * B[11] * B[16] * B[20] * B[24] * B[33] +
               B[1] * B[11] * B[16] * B[20] * B[27] * B[30] + B[1] * B[11] * B[16] * B[21] * B[24] * B[32] - B[1] * B[11] * B[16] * B[21] * B[26] * B[30] +
               B[2] * B[6] * B[13] * B[21] * B[28] * B[35] - B[2] * B[6] * B[13] * B[21] * B[29] * B[34] - B[2] * B[6] * B[13] * B[22] * B[27] * B[35] +
               B[2] * B[6] * B[13] * B[22] * B[29] * B[33] + B[2] * B[6] * B[13] * B[23] * B[27] * B[34] - B[2] * B[6] * B[13] * B[23] * B[28] * B[33] -
               B[2] * B[6] * B[15] * B[19] * B[28] * B[35] + B[2] * B[6] * B[15] * B[19] * B[29] * B[34] + B[2] * B[6] * B[15] * B[22] * B[25] * B[35] -
               B[2] * B[6] * B[15] * B[22] * B[29] * B[31] - B[2] * B[6] * B[15] * B[23] * B[25] * B[34] + B[2] * B[6] * B[15] * B[23] * B[28] * B[31] +
               B[2] * B[6] * B[16] * B[19] * B[27] * B[35] - B[2] * B[6] * B[16] * B[19] * B[29] * B[33] - B[2] * B[6] * B[16] * B[21] * B[25] * B[35] +
               B[2] * B[6] * B[16] * B[21] * B[29] * B[31] + B[2] * B[6] * B[16] * B[23] * B[25] * B[33] - B[2] * B[6] * B[16] * B[23] * B[27] * B[31] -
               B[2] * B[6] * B[17] * B[19] * B[27] * B[34] + B[2] * B[6] * B[17] * B[19] * B[28] * B[33] + B[2] * B[6] * B[17] * B[21] * B[25] * B[34] -
               B[2] * B[6] * B[17] * B[21] * B[28] * B[31] - B[2] * B[6] * B[17] * B[22] * B[25] * B[33] + B[2] * B[6] * B[17] * B[22] * B[27] * B[31] -
               B[2] * B[7] * B[12] * B[21] * B[28] * B[35] + B[2] * B[7] * B[12] * B[21] * B[29] * B[34] + B[2] * B[7] * B[12] * B[22] * B[27] * B[35] -
               B[2] * B[7] * B[12] * B[22] * B[29] * B[33] - B[2] * B[7] * B[12] * B[23] * B[27] * B[34] + B[2] * B[7] * B[12] * B[23] * B[28] * B[33] +
               B[2] * B[7] * B[15] * B[18] * B[28] * B[35] - B[2] * B[7] * B[15] * B[18] * B[29] * B[34] - B[2] * B[7] * B[15] * B[22] * B[24] * B[35] +
               B[2] * B[7] * B[15] * B[22] * B[29] * B[30] + B[2] * B[7] * B[15] * B[23] * B[24] * B[34] - B[2] * B[7] * B[15] * B[23] * B[28] * B[30] -
               B[2] * B[7] * B[16] * B[18] * B[27] * B[35] + B[2] * B[7] * B[16] * B[18] * B[29] * B[33] + B[2] * B[7] * B[16] * B[21] * B[24] * B[35] -
               B[2] * B[7] * B[16] * B[21] * B[29] * B[30] - B[2] * B[7] * B[16] * B[23] * B[24] * B[33] + B[2] * B[7] * B[16] * B[23] * B[27] * B[30] +
               B[2] * B[7] * B[17] * B[18] * B[27] * B[34] - B[2] * B[7] * B[17] * B[18] * B[28] * B[33] - B[2] * B[7] * B[17] * B[21] * B[24] * B[34] +
               B[2] * B[7] * B[17] * B[21] * B[28] * B[30] + B[2] * B[7] * B[17] * B[22] * B[24] * B[33] - B[2] * B[7] * B[17] * B[22] * B[27] * B[30] +
               B[2] * B[9] * B[12] * B[19] * B[28] * B[35] - B[2] * B[9] * B[12] * B[19] * B[29] * B[34] - B[2] * B[9] * B[12] * B[22] * B[25] * B[35] +
               B[2] * B[9] * B[12] * B[22] * B[29] * B[31] + B[2] * B[9] * B[12] * B[23] * B[25] * B[34] - B[2] * B[9] * B[12] * B[23] * B[28] * B[31] -
               B[2] * B[9] * B[13] * B[18] * B[28] * B[35] + B[2] * B[9] * B[13] * B[18] * B[29] * B[34] + B[2] * B[9] * B[13] * B[22] * B[24] * B[35] -
               B[2] * B[9] * B[13] * B[22] * B[29] * B[30] - B[2] * B[9] * B[13] * B[23] * B[24] * B[34] + B[2] * B[9] * B[13] * B[23] * B[28] * B[30] +
               B[2] * B[9] * B[16] * B[18] * B[25] * B[35] - B[2] * B[9] * B[16] * B[18] * B[29] * B[31] - B[2] * B[9] * B[16] * B[19] * B[24] * B[35] +
               B[2] * B[9] * B[16] * B[19] * B[29] * B[30] + B[2] * B[9] * B[16] * B[23] * B[24] * B[31] - B[2] * B[9] * B[16] * B[23] * B[25] * B[30] -
               B[2] * B[9] * B[17] * B[18] * B[25] * B[34] + B[2] * B[9] * B[17] * B[18] * B[28] * B[31] + B[2] * B[9] * B[17] * B[19] * B[24] * B[34] -
               B[2] * B[9] * B[17] * B[19] * B[28] * B[30] - B[2] * B[9] * B[17] * B[22] * B[24] * B[31] + B[2] * B[9] * B[17] * B[22] * B[25] * B[30] -
               B[2] * B[10] * B[12] * B[19] * B[27] * B[35] + B[2] * B[10] * B[12] * B[19] * B[29] * B[33] + B[2] * B[10] * B[12] * B[21] * B[25] * B[35] -
               B[2] * B[10] * B[12] * B[21] * B[29] * B[31] - B[2] * B[10] * B[12] * B[23] * B[25] * B[33] + B[2] * B[10] * B[12] * B[23] * B[27] * B[31] +
               B[2] * B[10] * B[13] * B[18] * B[27] * B[35] - B[2] * B[10] * B[13] * B[18] * B[29] * B[33] - B[2] * B[10] * B[13] * B[21] * B[24] * B[35] +
               B[2] * B[10] * B[13] * B[21] * B[29] * B[30] + B[2] * B[10] * B[13] * B[23] * B[24] * B[33] - B[2] * B[10] * B[13] * B[23] * B[27] * B[30] -
               B[2] * B[10] * B[15] * B[18] * B[25] * B[35] + B[2] * B[10] * B[15] * B[18] * B[29] * B[31] + B[2] * B[10] * B[15] * B[19] * B[24] * B[35] -
               B[2] * B[10] * B[15] * B[19] * B[29] * B[30] - B[2] * B[10] * B[15] * B[23] * B[24] * B[31] + B[2] * B[10] * B[15] * B[23] * B[25] * B[30] +
               B[2] * B[10] * B[17] * B[18] * B[25] * B[33] - B[2] * B[10] * B[17] * B[18] * B[27] * B[31] - B[2] * B[10] * B[17] * B[19] * B[24] * B[33] +
               B[2] * B[10] * B[17] * B[19] * B[27] * B[30] + B[2] * B[10] * B[17] * B[21] * B[24] * B[31] - B[2] * B[10] * B[17] * B[21] * B[25] * B[30] +
               B[2] * B[11] * B[12] * B[19] * B[27] * B[34] - B[2] * B[11] * B[12] * B[19] * B[28] * B[33] - B[2] * B[11] * B[12] * B[21] * B[25] * B[34] +
               B[2] * B[11] * B[12] * B[21] * B[28] * B[31] + B[2] * B[11] * B[12] * B[22] * B[25] * B[33] - B[2] * B[11] * B[12] * B[22] * B[27] * B[31] -
               B[2] * B[11] * B[13] * B[18] * B[27] * B[34] + B[2] * B[11] * B[13] * B[18] * B[28] * B[33] + B[2] * B[11] * B[13] * B[21] * B[24] * B[34] -
               B[2] * B[11] * B[13] * B[21] * B[28] * B[30] - B[2] * B[11] * B[13] * B[22] * B[24] * B[33] + B[2] * B[11] * B[13] * B[22] * B[27] * B[30] +
               B[2] * B[11] * B[15] * B[18] * B[25] * B[34] - B[2] * B[11] * B[15] * B[18] * B[28] * B[31] - B[2] * B[11] * B[15] * B[19] * B[24] * B[34] +
               B[2] * B[11] * B[15] * B[19] * B[28] * B[30] + B[2] * B[11] * B[15] * B[22] * B[24] * B[31] - B[2] * B[11] * B[15] * B[22] * B[25] * B[30] -
               B[2] * B[11] * B[16] * B[18] * B[25] * B[33] + B[2] * B[11] * B[16] * B[18] * B[27] * B[31] + B[2] * B[11] * B[16] * B[19] * B[24] * B[33] -
               B[2] * B[11] * B[16] * B[19] * B[27] * B[30] - B[2] * B[11] * B[16] * B[21] * B[24] * B[31] + B[2] * B[11] * B[16] * B[21] * B[25] * B[30] -
               B[3] * B[6] * B[13] * B[20] * B[28] * B[35] + B[3] * B[6] * B[13] * B[20] * B[29] * B[34] + B[3] * B[6] * B[13] * B[22] * B[26] * B[35] -
               B[3] * B[6] * B[13] * B[22] * B[29] * B[32] - B[3] * B[6] * B[13] * B[23] * B[26] * B[34] + B[3] * B[6] * B[13] * B[23] * B[28] * B[32] +
               B[3] * B[6] * B[14] * B[19] * B[28] * B[35] - B[3] * B[6] * B[14] * B[19] * B[29] * B[34] - B[3] * B[6] * B[14] * B[22] * B[25] * B[35] +
               B[3] * B[6] * B[14] * B[22] * B[29] * B[31] + B[3] * B[6] * B[14] * B[23] * B[25] * B[34] - B[3] * B[6] * B[14] * B[23] * B[28] * B[31] -
               B[3] * B[6] * B[16] * B[19] * B[26] * B[35] + B[3] * B[6] * B[16] * B[19] * B[29] * B[32] + B[3] * B[6] * B[16] * B[20] * B[25] * B[35] -
               B[3] * B[6] * B[16] * B[20] * B[29] * B[31] - B[3] * B[6] * B[16] * B[23] * B[25] * B[32] + B[3] * B[6] * B[16] * B[23] * B[26] * B[31] +
               B[3] * B[6] * B[17] * B[19] * B[26] * B[34] - B[3] * B[6] * B[17] * B[19] * B[28] * B[32] - B[3] * B[6] * B[17] * B[20] * B[25] * B[34] +
               B[3] * B[6] * B[17] * B[20] * B[28] * B[31] + B[3] * B[6] * B[17] * B[22] * B[25] * B[32] - B[3] * B[6] * B[17] * B[22] * B[26] * B[31] +
               B[3] * B[7] * B[12] * B[20] * B[28] * B[35] - B[3] * B[7] * B[12] * B[20] * B[29] * B[34] - B[3] * B[7] * B[12] * B[22] * B[26] * B[35] +
               B[3] * B[7] * B[12] * B[22] * B[29] * B[32] + B[3] * B[7] * B[12] * B[23] * B[26] * B[34] - B[3] * B[7] * B[12] * B[23] * B[28] * B[32] -
               B[3] * B[7] * B[14] * B[18] * B[28] * B[35] + B[3] * B[7] * B[14] * B[18] * B[29] * B[34] + B[3] * B[7] * B[14] * B[22] * B[24] * B[35] -
               B[3] * B[7] * B[14] * B[22] * B[29] * B[30] - B[3] * B[7] * B[14] * B[23] * B[24] * B[34] + B[3] * B[7] * B[14] * B[23] * B[28] * B[30] +
               B[3] * B[7] * B[16] * B[18] * B[26] * B[35] - B[3] * B[7] * B[16] * B[18] * B[29] * B[32] - B[3] * B[7] * B[16] * B[20] * B[24] * B[35] +
               B[3] * B[7] * B[16] * B[20] * B[29] * B[30] + B[3] * B[7] * B[16] * B[23] * B[24] * B[32] - B[3] * B[7] * B[16] * B[23] * B[26] * B[30] -
               B[3] * B[7] * B[17] * B[18] * B[26] * B[34] + B[3] * B[7] * B[17] * B[18] * B[28] * B[32] + B[3] * B[7] * B[17] * B[20] * B[24] * B[34] -
               B[3] * B[7] * B[17] * B[20] * B[28] * B[30] - B[3] * B[7] * B[17] * B[22] * B[24] * B[32] + B[3] * B[7] * B[17] * B[22] * B[26] * B[30] -
               B[3] * B[8] * B[12] * B[19] * B[28] * B[35] + B[3] * B[8] * B[12] * B[19] * B[29] * B[34] + B[3] * B[8] * B[12] * B[22] * B[25] * B[35] -
               B[3] * B[8] * B[12] * B[22] * B[29] * B[31] - B[3] * B[8] * B[12] * B[23] * B[25] * B[34] + B[3] * B[8] * B[12] * B[23] * B[28] * B[31] +
               B[3] * B[8] * B[13] * B[18] * B[28] * B[35] - B[3] * B[8] * B[13] * B[18] * B[29] * B[34] - B[3] * B[8] * B[13] * B[22] * B[24] * B[35] +
               B[3] * B[8] * B[13] * B[22] * B[29] * B[30] + B[3] * B[8] * B[13] * B[23] * B[24] * B[34] - B[3] * B[8] * B[13] * B[23] * B[28] * B[30] -
               B[3] * B[8] * B[16] * B[18] * B[25] * B[35] + B[3] * B[8] * B[16] * B[18] * B[29] * B[31] + B[3] * B[8] * B[16] * B[19] * B[24] * B[35] -
               B[3] * B[8] * B[16] * B[19] * B[29] * B[30] - B[3] * B[8] * B[16] * B[23] * B[24] * B[31] + B[3] * B[8] * B[16] * B[23] * B[25] * B[30] +
               B[3] * B[8] * B[17] * B[18] * B[25] * B[34] - B[3] * B[8] * B[17] * B[18] * B[28] * B[31] - B[3] * B[8] * B[17] * B[19] * B[24] * B[34] +
               B[3] * B[8] * B[17] * B[19] * B[28] * B[30] + B[3] * B[8] * B[17] * B[22] * B[24] * B[31] - B[3] * B[8] * B[17] * B[22] * B[25] * B[30] +
               B[3] * B[10] * B[12] * B[19] * B[26] * B[35] - B[3] * B[10] * B[12] * B[19] * B[29] * B[32] - B[3] * B[10] * B[12] * B[20] * B[25] * B[35] +
               B[3] * B[10] * B[12] * B[20] * B[29] * B[31] + B[3] * B[10] * B[12] * B[23] * B[25] * B[32] - B[3] * B[10] * B[12] * B[23] * B[26] * B[31] -
               B[3] * B[10] * B[13] * B[18] * B[26] * B[35] + B[3] * B[10] * B[13] * B[18] * B[29] * B[32] + B[3] * B[10] * B[13] * B[20] * B[24] * B[35] -
               B[3] * B[10] * B[13] * B[20] * B[29] * B[30] - B[3] * B[10] * B[13] * B[23] * B[24] * B[32] + B[3] * B[10] * B[13] * B[23] * B[26] * B[30] +
               B[3] * B[10] * B[14] * B[18] * B[25] * B[35] - B[3] * B[10] * B[14] * B[18] * B[29] * B[31] - B[3] * B[10] * B[14] * B[19] * B[24] * B[35] +
               B[3] * B[10] * B[14] * B[19] * B[29] * B[30] + B[3] * B[10] * B[14] * B[23] * B[24] * B[31] - B[3] * B[10] * B[14] * B[23] * B[25] * B[30] -
               B[3] * B[10] * B[17] * B[18] * B[25] * B[32] + B[3] * B[10] * B[17] * B[18] * B[26] * B[31] + B[3] * B[10] * B[17] * B[19] * B[24] * B[32] -
               B[3] * B[10] * B[17] * B[19] * B[26] * B[30] - B[3] * B[10] * B[17] * B[20] * B[24] * B[31] + B[3] * B[10] * B[17] * B[20] * B[25] * B[30] -
               B[3] * B[11] * B[12] * B[19] * B[26] * B[34] + B[3] * B[11] * B[12] * B[19] * B[28] * B[32] + B[3] * B[11] * B[12] * B[20] * B[25] * B[34] -
               B[3] * B[11] * B[12] * B[20] * B[28] * B[31] - B[3] * B[11] * B[12] * B[22] * B[25] * B[32] + B[3] * B[11] * B[12] * B[22] * B[26] * B[31] +
               B[3] * B[11] * B[13] * B[18] * B[26] * B[34] - B[3] * B[11] * B[13] * B[18] * B[28] * B[32] - B[3] * B[11] * B[13] * B[20] * B[24] * B[34] +
               B[3] * B[11] * B[13] * B[20] * B[28] * B[30] + B[3] * B[11] * B[13] * B[22] * B[24] * B[32] - B[3] * B[11] * B[13] * B[22] * B[26] * B[30] -
               B[3] * B[11] * B[14] * B[18] * B[25] * B[34] + B[3] * B[11] * B[14] * B[18] * B[28] * B[31] + B[3] * B[11] * B[14] * B[19] * B[24] * B[34] -
               B[3] * B[11] * B[14] * B[19] * B[28] * B[30] - B[3] * B[11] * B[14] * B[22] * B[24] * B[31] + B[3] * B[11] * B[14] * B[22] * B[25] * B[30] +
               B[3] * B[11] * B[16] * B[18] * B[25] * B[32] - B[3] * B[11] * B[16] * B[18] * B[26] * B[31] - B[3] * B[11] * B[16] * B[19] * B[24] * B[32] +
               B[3] * B[11] * B[16] * B[19] * B[26] * B[30] + B[3] * B[11] * B[16] * B[20] * B[24] * B[31] - B[3] * B[11] * B[16] * B[20] * B[25] * B[30] +
               B[4] * B[6] * B[13] * B[20] * B[27] * B[35] - B[4] * B[6] * B[13] * B[20] * B[29] * B[33] - B[4] * B[6] * B[13] * B[21] * B[26] * B[35] +
               B[4] * B[6] * B[13] * B[21] * B[29] * B[32] + B[4] * B[6] * B[13] * B[23] * B[26] * B[33] - B[4] * B[6] * B[13] * B[23] * B[27] * B[32] -
               B[4] * B[6] * B[14] * B[19] * B[27] * B[35] + B[4] * B[6] * B[14] * B[19] * B[29] * B[33] + B[4] * B[6] * B[14] * B[21] * B[25] * B[35] -
               B[4] * B[6] * B[14] * B[21] * B[29] * B[31] - B[4] * B[6] * B[14] * B[23] * B[25] * B[33] + B[4] * B[6] * B[14] * B[23] * B[27] * B[31] +
               B[4] * B[6] * B[15] * B[19] * B[26] * B[35] - B[4] * B[6] * B[15] * B[19] * B[29] * B[32] - B[4] * B[6] * B[15] * B[20] * B[25] * B[35] +
               B[4] * B[6] * B[15] * B[20] * B[29] * B[31] + B[4] * B[6] * B[15] * B[23] * B[25] * B[32] - B[4] * B[6] * B[15] * B[23] * B[26] * B[31] -
               B[4] * B[6] * B[17] * B[19] * B[26] * B[33] + B[4] * B[6] * B[17] * B[19] * B[27] * B[32] + B[4] * B[6] * B[17] * B[20] * B[25] * B[33] -
               B[4] * B[6] * B[17] * B[20] * B[27] * B[31] - B[4] * B[6] * B[17] * B[21] * B[25] * B[32] + B[4] * B[6] * B[17] * B[21] * B[26] * B[31] -
               B[4] * B[7] * B[12] * B[20] * B[27] * B[35] + B[4] * B[7] * B[12] * B[20] * B[29] * B[33] + B[4] * B[7] * B[12] * B[21] * B[26] * B[35] -
               B[4] * B[7] * B[12] * B[21] * B[29] * B[32] - B[4] * B[7] * B[12] * B[23] * B[26] * B[33] + B[4] * B[7] * B[12] * B[23] * B[27] * B[32] +
               B[4] * B[7] * B[14] * B[18] * B[27] * B[35] - B[4] * B[7] * B[14] * B[18] * B[29] * B[33] - B[4] * B[7] * B[14] * B[21] * B[24] * B[35] +
               B[4] * B[7] * B[14] * B[21] * B[29] * B[30] + B[4] * B[7] * B[14] * B[23] * B[24] * B[33] - B[4] * B[7] * B[14] * B[23] * B[27] * B[30] -
               B[4] * B[7] * B[15] * B[18] * B[26] * B[35] + B[4] * B[7] * B[15] * B[18] * B[29] * B[32] + B[4] * B[7] * B[15] * B[20] * B[24] * B[35] -
               B[4] * B[7] * B[15] * B[20] * B[29] * B[30] - B[4] * B[7] * B[15] * B[23] * B[24] * B[32] + B[4] * B[7] * B[15] * B[23] * B[26] * B[30] +
               B[4] * B[7] * B[17] * B[18] * B[26] * B[33] - B[4] * B[7] * B[17] * B[18] * B[27] * B[32] - B[4] * B[7] * B[17] * B[20] * B[24] * B[33] +
               B[4] * B[7] * B[17] * B[20] * B[27] * B[30] + B[4] * B[7] * B[17] * B[21] * B[24] * B[32] - B[4] * B[7] * B[17] * B[21] * B[26] * B[30] +
               B[4] * B[8] * B[12] * B[19] * B[27] * B[35] - B[4] * B[8] * B[12] * B[19] * B[29] * B[33] - B[4] * B[8] * B[12] * B[21] * B[25] * B[35] +
               B[4] * B[8] * B[12] * B[21] * B[29] * B[31] + B[4] * B[8] * B[12] * B[23] * B[25] * B[33] - B[4] * B[8] * B[12] * B[23] * B[27] * B[31] -
               B[4] * B[8] * B[13] * B[18] * B[27] * B[35] + B[4] * B[8] * B[13] * B[18] * B[29] * B[33] + B[4] * B[8] * B[13] * B[21] * B[24] * B[35] -
               B[4] * B[8] * B[13] * B[21] * B[29] * B[30] - B[4] * B[8] * B[13] * B[23] * B[24] * B[33] + B[4] * B[8] * B[13] * B[23] * B[27] * B[30] +
               B[4] * B[8] * B[15] * B[18] * B[25] * B[35] - B[4] * B[8] * B[15] * B[18] * B[29] * B[31] - B[4] * B[8] * B[15] * B[19] * B[24] * B[35] +
               B[4] * B[8] * B[15] * B[19] * B[29] * B[30] + B[4] * B[8] * B[15] * B[23] * B[24] * B[31] - B[4] * B[8] * B[15] * B[23] * B[25] * B[30] -
               B[4] * B[8] * B[17] * B[18] * B[25] * B[33] + B[4] * B[8] * B[17] * B[18] * B[27] * B[31] + B[4] * B[8] * B[17] * B[19] * B[24] * B[33] -
               B[4] * B[8] * B[17] * B[19] * B[27] * B[30] - B[4] * B[8] * B[17] * B[21] * B[24] * B[31] + B[4] * B[8] * B[17] * B[21] * B[25] * B[30] -
               B[4] * B[9] * B[12] * B[19] * B[26] * B[35] + B[4] * B[9] * B[12] * B[19] * B[29] * B[32] + B[4] * B[9] * B[12] * B[20] * B[25] * B[35] -
               B[4] * B[9] * B[12] * B[20] * B[29] * B[31] - B[4] * B[9] * B[12] * B[23] * B[25] * B[32] + B[4] * B[9] * B[12] * B[23] * B[26] * B[31] +
               B[4] * B[9] * B[13] * B[18] * B[26] * B[35] - B[4] * B[9] * B[13] * B[18] * B[29] * B[32] - B[4] * B[9] * B[13] * B[20] * B[24] * B[35] +
               B[4] * B[9] * B[13] * B[20] * B[29] * B[30] + B[4] * B[9] * B[13] * B[23] * B[24] * B[32] - B[4] * B[9] * B[13] * B[23] * B[26] * B[30] -
               B[4] * B[9] * B[14] * B[18] * B[25] * B[35] + B[4] * B[9] * B[14] * B[18] * B[29] * B[31] + B[4] * B[9] * B[14] * B[19] * B[24] * B[35] -
               B[4] * B[9] * B[14] * B[19] * B[29] * B[30] - B[4] * B[9] * B[14] * B[23] * B[24] * B[31] + B[4] * B[9] * B[14] * B[23] * B[25] * B[30] +
               B[4] * B[9] * B[17] * B[18] * B[25] * B[32] - B[4] * B[9] * B[17] * B[18] * B[26] * B[31] - B[4] * B[9] * B[17] * B[19] * B[24] * B[32] +
               B[4] * B[9] * B[17] * B[19] * B[26] * B[30] + B[4] * B[9] * B[17] * B[20] * B[24] * B[31] - B[4] * B[9] * B[17] * B[20] * B[25] * B[30] +
               B[4] * B[11] * B[12] * B[19] * B[26] * B[33] - B[4] * B[11] * B[12] * B[19] * B[27] * B[32] - B[4] * B[11] * B[12] * B[20] * B[25] * B[33] +
               B[4] * B[11] * B[12] * B[20] * B[27] * B[31] + B[4] * B[11] * B[12] * B[21] * B[25] * B[32] - B[4] * B[11] * B[12] * B[21] * B[26] * B[31] -
               B[4] * B[11] * B[13] * B[18] * B[26] * B[33] + B[4] * B[11] * B[13] * B[18] * B[27] * B[32] + B[4] * B[11] * B[13] * B[20] * B[24] * B[33] -
               B[4] * B[11] * B[13] * B[20] * B[27] * B[30] - B[4] * B[11] * B[13] * B[21] * B[24] * B[32] + B[4] * B[11] * B[13] * B[21] * B[26] * B[30] +
               B[4] * B[11] * B[14] * B[18] * B[25] * B[33] - B[4] * B[11] * B[14] * B[18] * B[27] * B[31] - B[4] * B[11] * B[14] * B[19] * B[24] * B[33] +
               B[4] * B[11] * B[14] * B[19] * B[27] * B[30] + B[4] * B[11] * B[14] * B[21] * B[24] * B[31] - B[4] * B[11] * B[14] * B[21] * B[25] * B[30] -
               B[4] * B[11] * B[15] * B[18] * B[25] * B[32] + B[4] * B[11] * B[15] * B[18] * B[26] * B[31] + B[4] * B[11] * B[15] * B[19] * B[24] * B[32] -
               B[4] * B[11] * B[15] * B[19] * B[26] * B[30] - B[4] * B[11] * B[15] * B[20] * B[24] * B[31] + B[4] * B[11] * B[15] * B[20] * B[25] * B[30] -
               B[5] * B[6] * B[13] * B[20] * B[27] * B[34] + B[5] * B[6] * B[13] * B[20] * B[28] * B[33] + B[5] * B[6] * B[13] * B[21] * B[26] * B[34] -
               B[5] * B[6] * B[13] * B[21] * B[28] * B[32] - B[5] * B[6] * B[13] * B[22] * B[26] * B[33] + B[5] * B[6] * B[13] * B[22] * B[27] * B[32] +
               B[5] * B[6] * B[14] * B[19] * B[27] * B[34] - B[5] * B[6] * B[14] * B[19] * B[28] * B[33] - B[5] * B[6] * B[14] * B[21] * B[25] * B[34] +
               B[5] * B[6] * B[14] * B[21] * B[28] * B[31] + B[5] * B[6] * B[14] * B[22] * B[25] * B[33] - B[5] * B[6] * B[14] * B[22] * B[27] * B[31] -
               B[5] * B[6] * B[15] * B[19] * B[26] * B[34] + B[5] * B[6] * B[15] * B[19] * B[28] * B[32] + B[5] * B[6] * B[15] * B[20] * B[25] * B[34] -
               B[5] * B[6] * B[15] * B[20] * B[28] * B[31] - B[5] * B[6] * B[15] * B[22] * B[25] * B[32] + B[5] * B[6] * B[15] * B[22] * B[26] * B[31] +
               B[5] * B[6] * B[16] * B[19] * B[26] * B[33] - B[5] * B[6] * B[16] * B[19] * B[27] * B[32] - B[5] * B[6] * B[16] * B[20] * B[25] * B[33] +
               B[5] * B[6] * B[16] * B[20] * B[27] * B[31] + B[5] * B[6] * B[16] * B[21] * B[25] * B[32] - B[5] * B[6] * B[16] * B[21] * B[26] * B[31] +
               B[5] * B[7] * B[12] * B[20] * B[27] * B[34] - B[5] * B[7] * B[12] * B[20] * B[28] * B[33] - B[5] * B[7] * B[12] * B[21] * B[26] * B[34] +
               B[5] * B[7] * B[12] * B[21] * B[28] * B[32] + B[5] * B[7] * B[12] * B[22] * B[26] * B[33] - B[5] * B[7] * B[12] * B[22] * B[27] * B[32] -
               B[5] * B[7] * B[14] * B[18] * B[27] * B[34] + B[5] * B[7] * B[14] * B[18] * B[28] * B[33] + B[5] * B[7] * B[14] * B[21] * B[24] * B[34] -
               B[5] * B[7] * B[14] * B[21] * B[28] * B[30] - B[5] * B[7] * B[14] * B[22] * B[24] * B[33] + B[5] * B[7] * B[14] * B[22] * B[27] * B[30] +
               B[5] * B[7] * B[15] * B[18] * B[26] * B[34] - B[5] * B[7] * B[15] * B[18] * B[28] * B[32] - B[5] * B[7] * B[15] * B[20] * B[24] * B[34] +
               B[5] * B[7] * B[15] * B[20] * B[28] * B[30] + B[5] * B[7] * B[15] * B[22] * B[24] * B[32] - B[5] * B[7] * B[15] * B[22] * B[26] * B[30] -
               B[5] * B[7] * B[16] * B[18] * B[26] * B[33] + B[5] * B[7] * B[16] * B[18] * B[27] * B[32] + B[5] * B[7] * B[16] * B[20] * B[24] * B[33] -
               B[5] * B[7] * B[16] * B[20] * B[27] * B[30] - B[5] * B[7] * B[16] * B[21] * B[24] * B[32] + B[5] * B[7] * B[16] * B[21] * B[26] * B[30] -
               B[5] * B[8] * B[12] * B[19] * B[27] * B[34] + B[5] * B[8] * B[12] * B[19] * B[28] * B[33] + B[5] * B[8] * B[12] * B[21] * B[25] * B[34] -
               B[5] * B[8] * B[12] * B[21] * B[28] * B[31] - B[5] * B[8] * B[12] * B[22] * B[25] * B[33] + B[5] * B[8] * B[12] * B[22] * B[27] * B[31] +
               B[5] * B[8] * B[13] * B[18] * B[27] * B[34] - B[5] * B[8] * B[13] * B[18] * B[28] * B[33] - B[5] * B[8] * B[13] * B[21] * B[24] * B[34] +
               B[5] * B[8] * B[13] * B[21] * B[28] * B[30] + B[5] * B[8] * B[13] * B[22] * B[24] * B[33] - B[5] * B[8] * B[13] * B[22] * B[27] * B[30] -
               B[5] * B[8] * B[15] * B[18] * B[25] * B[34] + B[5] * B[8] * B[15] * B[18] * B[28] * B[31] + B[5] * B[8] * B[15] * B[19] * B[24] * B[34] -
               B[5] * B[8] * B[15] * B[19] * B[28] * B[30] - B[5] * B[8] * B[15] * B[22] * B[24] * B[31] + B[5] * B[8] * B[15] * B[22] * B[25] * B[30] +
               B[5] * B[8] * B[16] * B[18] * B[25] * B[33] - B[5] * B[8] * B[16] * B[18] * B[27] * B[31] - B[5] * B[8] * B[16] * B[19] * B[24] * B[33] +
               B[5] * B[8] * B[16] * B[19] * B[27] * B[30] + B[5] * B[8] * B[16] * B[21] * B[24] * B[31] - B[5] * B[8] * B[16] * B[21] * B[25] * B[30] +
               B[5] * B[9] * B[12] * B[19] * B[26] * B[34] - B[5] * B[9] * B[12] * B[19] * B[28] * B[32] - B[5] * B[9] * B[12] * B[20] * B[25] * B[34] +
               B[5] * B[9] * B[12] * B[20] * B[28] * B[31] + B[5] * B[9] * B[12] * B[22] * B[25] * B[32] - B[5] * B[9] * B[12] * B[22] * B[26] * B[31] -
               B[5] * B[9] * B[13] * B[18] * B[26] * B[34] + B[5] * B[9] * B[13] * B[18] * B[28] * B[32] + B[5] * B[9] * B[13] * B[20] * B[24] * B[34] -
               B[5] * B[9] * B[13] * B[20] * B[28] * B[30] - B[5] * B[9] * B[13] * B[22] * B[24] * B[32] + B[5] * B[9] * B[13] * B[22] * B[26] * B[30] +
               B[5] * B[9] * B[14] * B[18] * B[25] * B[34] - B[5] * B[9] * B[14] * B[18] * B[28] * B[31] - B[5] * B[9] * B[14] * B[19] * B[24] * B[34] +
               B[5] * B[9] * B[14] * B[19] * B[28] * B[30] + B[5] * B[9] * B[14] * B[22] * B[24] * B[31] - B[5] * B[9] * B[14] * B[22] * B[25] * B[30] -
               B[5] * B[9] * B[16] * B[18] * B[25] * B[32] + B[5] * B[9] * B[16] * B[18] * B[26] * B[31] + B[5] * B[9] * B[16] * B[19] * B[24] * B[32] -
               B[5] * B[9] * B[16] * B[19] * B[26] * B[30] - B[5] * B[9] * B[16] * B[20] * B[24] * B[31] + B[5] * B[9] * B[16] * B[20] * B[25] * B[30] -
               B[5] * B[10] * B[12] * B[19] * B[26] * B[33] + B[5] * B[10] * B[12] * B[19] * B[27] * B[32] + B[5] * B[10] * B[12] * B[20] * B[25] * B[33] -
               B[5] * B[10] * B[12] * B[20] * B[27] * B[31] - B[5] * B[10] * B[12] * B[21] * B[25] * B[32] + B[5] * B[10] * B[12] * B[21] * B[26] * B[31] +
               B[5] * B[10] * B[13] * B[18] * B[26] * B[33] - B[5] * B[10] * B[13] * B[18] * B[27] * B[32] - B[5] * B[10] * B[13] * B[20] * B[24] * B[33] +
               B[5] * B[10] * B[13] * B[20] * B[27] * B[30] + B[5] * B[10] * B[13] * B[21] * B[24] * B[32] - B[5] * B[10] * B[13] * B[21] * B[26] * B[30] -
               B[5] * B[10] * B[14] * B[18] * B[25] * B[33] + B[5] * B[10] * B[14] * B[18] * B[27] * B[31] + B[5] * B[10] * B[14] * B[19] * B[24] * B[33] -
               B[5] * B[10] * B[14] * B[19] * B[27] * B[30] - B[5] * B[10] * B[14] * B[21] * B[24] * B[31] + B[5] * B[10] * B[14] * B[21] * B[25] * B[30] +
               B[5] * B[10] * B[15] * B[18] * B[25] * B[32] - B[5] * B[10] * B[15] * B[18] * B[26] * B[31] - B[5] * B[10] * B[15] * B[19] * B[24] * B[32] +
               B[5] * B[10] * B[15] * B[19] * B[26] * B[30] + B[5] * B[10] * B[15] * B[20] * B[24] * B[31] - B[5] * B[10] * B[15] * B[20] * B[25] * B[30];

    L[0] = (B[1] * B[8] * B[15] * B[22] * B[29] - B[1] * B[8] * B[15] * B[23] * B[28] - B[1] * B[8] * B[16] * B[21] * B[29] + B[1] * B[8] * B[16] * B[23] * B[27] +
            B[1] * B[8] * B[17] * B[21] * B[28] - B[1] * B[8] * B[17] * B[22] * B[27] - B[1] * B[9] * B[14] * B[22] * B[29] + B[1] * B[9] * B[14] * B[23] * B[28] +
            B[1] * B[9] * B[16] * B[20] * B[29] - B[1] * B[9] * B[16] * B[23] * B[26] - B[1] * B[9] * B[17] * B[20] * B[28] + B[1] * B[9] * B[17] * B[22] * B[26] +
            B[1] * B[10] * B[14] * B[21] * B[29] - B[1] * B[10] * B[14] * B[23] * B[27] - B[1] * B[10] * B[15] * B[20] * B[29] + B[1] * B[10] * B[15] * B[23] * B[26] +
            B[1] * B[10] * B[17] * B[20] * B[27] - B[1] * B[10] * B[17] * B[21] * B[26] - B[1] * B[11] * B[14] * B[21] * B[28] + B[1] * B[11] * B[14] * B[22] * B[27] +
            B[1] * B[11] * B[15] * B[20] * B[28] - B[1] * B[11] * B[15] * B[22] * B[26] - B[1] * B[11] * B[16] * B[20] * B[27] + B[1] * B[11] * B[16] * B[21] * B[26] -
            B[2] * B[7] * B[15] * B[22] * B[29] + B[2] * B[7] * B[15] * B[23] * B[28] + B[2] * B[7] * B[16] * B[21] * B[29] - B[2] * B[7] * B[16] * B[23] * B[27] -
            B[2] * B[7] * B[17] * B[21] * B[28] + B[2] * B[7] * B[17] * B[22] * B[27] + B[2] * B[9] * B[13] * B[22] * B[29] - B[2] * B[9] * B[13] * B[23] * B[28] -
            B[2] * B[9] * B[16] * B[19] * B[29] + B[2] * B[9] * B[16] * B[23] * B[25] + B[2] * B[9] * B[17] * B[19] * B[28] - B[2] * B[9] * B[17] * B[22] * B[25] -
            B[2] * B[10] * B[13] * B[21] * B[29] + B[2] * B[10] * B[13] * B[23] * B[27] + B[2] * B[10] * B[15] * B[19] * B[29] - B[2] * B[10] * B[15] * B[23] * B[25] -
            B[2] * B[10] * B[17] * B[19] * B[27] + B[2] * B[10] * B[17] * B[21] * B[25] + B[2] * B[11] * B[13] * B[21] * B[28] - B[2] * B[11] * B[13] * B[22] * B[27] -
            B[2] * B[11] * B[15] * B[19] * B[28] + B[2] * B[11] * B[15] * B[22] * B[25] + B[2] * B[11] * B[16] * B[19] * B[27] - B[2] * B[11] * B[16] * B[21] * B[25] +
            B[3] * B[7] * B[14] * B[22] * B[29] - B[3] * B[7] * B[14] * B[23] * B[28] - B[3] * B[7] * B[16] * B[20] * B[29] + B[3] * B[7] * B[16] * B[23] * B[26] +
            B[3] * B[7] * B[17] * B[20] * B[28] - B[3] * B[7] * B[17] * B[22] * B[26] - B[3] * B[8] * B[13] * B[22] * B[29] + B[3] * B[8] * B[13] * B[23] * B[28] +
            B[3] * B[8] * B[16] * B[19] * B[29] - B[3] * B[8] * B[16] * B[23] * B[25] - B[3] * B[8] * B[17] * B[19] * B[28] + B[3] * B[8] * B[17] * B[22] * B[25] +
            B[3] * B[10] * B[13] * B[20] * B[29] - B[3] * B[10] * B[13] * B[23] * B[26] - B[3] * B[10] * B[14] * B[19] * B[29] + B[3] * B[10] * B[14] * B[23] * B[25] +
            B[3] * B[10] * B[17] * B[19] * B[26] - B[3] * B[10] * B[17] * B[20] * B[25] - B[3] * B[11] * B[13] * B[20] * B[28] + B[3] * B[11] * B[13] * B[22] * B[26] +
            B[3] * B[11] * B[14] * B[19] * B[28] - B[3] * B[11] * B[14] * B[22] * B[25] - B[3] * B[11] * B[16] * B[19] * B[26] + B[3] * B[11] * B[16] * B[20] * B[25] -
            B[4] * B[7] * B[14] * B[21] * B[29] + B[4] * B[7] * B[14] * B[23] * B[27] + B[4] * B[7] * B[15] * B[20] * B[29] - B[4] * B[7] * B[15] * B[23] * B[26] -
            B[4] * B[7] * B[17] * B[20] * B[27] + B[4] * B[7] * B[17] * B[21] * B[26] + B[4] * B[8] * B[13] * B[21] * B[29] - B[4] * B[8] * B[13] * B[23] * B[27] -
            B[4] * B[8] * B[15] * B[19] * B[29] + B[4] * B[8] * B[15] * B[23] * B[25] + B[4] * B[8] * B[17] * B[19] * B[27] - B[4] * B[8] * B[17] * B[21] * B[25] -
            B[4] * B[9] * B[13] * B[20] * B[29] + B[4] * B[9] * B[13] * B[23] * B[26] + B[4] * B[9] * B[14] * B[19] * B[29] - B[4] * B[9] * B[14] * B[23] * B[25] -
            B[4] * B[9] * B[17] * B[19] * B[26] + B[4] * B[9] * B[17] * B[20] * B[25] + B[4] * B[11] * B[13] * B[20] * B[27] - B[4] * B[11] * B[13] * B[21] * B[26] -
            B[4] * B[11] * B[14] * B[19] * B[27] + B[4] * B[11] * B[14] * B[21] * B[25] + B[4] * B[11] * B[15] * B[19] * B[26] - B[4] * B[11] * B[15] * B[20] * B[25] +
            B[5] * B[7] * B[14] * B[21] * B[28] - B[5] * B[7] * B[14] * B[22] * B[27] - B[5] * B[7] * B[15] * B[20] * B[28] + B[5] * B[7] * B[15] * B[22] * B[26] +
            B[5] * B[7] * B[16] * B[20] * B[27] - B[5] * B[7] * B[16] * B[21] * B[26] - B[5] * B[8] * B[13] * B[21] * B[28] + B[5] * B[8] * B[13] * B[22] * B[27] +
            B[5] * B[8] * B[15] * B[19] * B[28] - B[5] * B[8] * B[15] * B[22] * B[25] - B[5] * B[8] * B[16] * B[19] * B[27] + B[5] * B[8] * B[16] * B[21] * B[25] +
            B[5] * B[9] * B[13] * B[20] * B[28] - B[5] * B[9] * B[13] * B[22] * B[26] - B[5] * B[9] * B[14] * B[19] * B[28] + B[5] * B[9] * B[14] * B[22] * B[25] +
            B[5] * B[9] * B[16] * B[19] * B[26] - B[5] * B[9] * B[16] * B[20] * B[25] - B[5] * B[10] * B[13] * B[20] * B[27] + B[5] * B[10] * B[13] * B[21] * B[26] +
            B[5] * B[10] * B[14] * B[19] * B[27] - B[5] * B[10] * B[14] * B[21] * B[25] - B[5] * B[10] * B[15] * B[19] * B[26] + B[5] * B[10] * B[15] * B[20] * B[25] +
            B[1] * B[8] * B[15] * B[28] * B[35] - B[1] * B[8] * B[15] * B[29] * B[34] - B[1] * B[8] * B[16] * B[27] * B[35] + B[1] * B[8] * B[16] * B[29] * B[33] +
            B[1] * B[8] * B[17] * B[27] * B[34] - B[1] * B[8] * B[17] * B[28] * B[33] - B[1] * B[9] * B[14] * B[28] * B[35] + B[1] * B[9] * B[14] * B[29] * B[34] +
            B[1] * B[9] * B[16] * B[26] * B[35] - B[1] * B[9] * B[16] * B[29] * B[32] - B[1] * B[9] * B[17] * B[26] * B[34] + B[1] * B[9] * B[17] * B[28] * B[32] +
            B[1] * B[10] * B[14] * B[27] * B[35] - B[1] * B[10] * B[14] * B[29] * B[33] - B[1] * B[10] * B[15] * B[26] * B[35] + B[1] * B[10] * B[15] * B[29] * B[32] +
            B[1] * B[10] * B[17] * B[26] * B[33] - B[1] * B[10] * B[17] * B[27] * B[32] - B[1] * B[11] * B[14] * B[27] * B[34] + B[1] * B[11] * B[14] * B[28] * B[33] +
            B[1] * B[11] * B[15] * B[26] * B[34] - B[1] * B[11] * B[15] * B[28] * B[32] - B[1] * B[11] * B[16] * B[26] * B[33] + B[1] * B[11] * B[16] * B[27] * B[32] -
            B[2] * B[7] * B[15] * B[28] * B[35] + B[2] * B[7] * B[15] * B[29] * B[34] + B[2] * B[7] * B[16] * B[27] * B[35] - B[2] * B[7] * B[16] * B[29] * B[33] -
            B[2] * B[7] * B[17] * B[27] * B[34] + B[2] * B[7] * B[17] * B[28] * B[33] + B[2] * B[9] * B[13] * B[28] * B[35] - B[2] * B[9] * B[13] * B[29] * B[34] -
            B[2] * B[9] * B[16] * B[25] * B[35] + B[2] * B[9] * B[16] * B[29] * B[31] + B[2] * B[9] * B[17] * B[25] * B[34] - B[2] * B[9] * B[17] * B[28] * B[31] -
            B[2] * B[10] * B[13] * B[27] * B[35] + B[2] * B[10] * B[13] * B[29] * B[33] + B[2] * B[10] * B[15] * B[25] * B[35] - B[2] * B[10] * B[15] * B[29] * B[31] -
            B[2] * B[10] * B[17] * B[25] * B[33] + B[2] * B[10] * B[17] * B[27] * B[31] + B[2] * B[11] * B[13] * B[27] * B[34] - B[2] * B[11] * B[13] * B[28] * B[33] -
            B[2] * B[11] * B[15] * B[25] * B[34] + B[2] * B[11] * B[15] * B[28] * B[31] + B[2] * B[11] * B[16] * B[25] * B[33] - B[2] * B[11] * B[16] * B[27] * B[31] +
            B[3] * B[7] * B[14] * B[28] * B[35] - B[3] * B[7] * B[14] * B[29] * B[34] - B[3] * B[7] * B[16] * B[26] * B[35] + B[3] * B[7] * B[16] * B[29] * B[32] +
            B[3] * B[7] * B[17] * B[26] * B[34] - B[3] * B[7] * B[17] * B[28] * B[32] - B[3] * B[8] * B[13] * B[28] * B[35] + B[3] * B[8] * B[13] * B[29] * B[34] +
            B[3] * B[8] * B[16] * B[25] * B[35] - B[3] * B[8] * B[16] * B[29] * B[31] - B[3] * B[8] * B[17] * B[25] * B[34] + B[3] * B[8] * B[17] * B[28] * B[31] +
            B[3] * B[10] * B[13] * B[26] * B[35] - B[3] * B[10] * B[13] * B[29] * B[32] - B[3] * B[10] * B[14] * B[25] * B[35] + B[3] * B[10] * B[14] * B[29] * B[31] +
            B[3] * B[10] * B[17] * B[25] * B[32] - B[3] * B[10] * B[17] * B[26] * B[31] - B[3] * B[11] * B[13] * B[26] * B[34] + B[3] * B[11] * B[13] * B[28] * B[32] +
            B[3] * B[11] * B[14] * B[25] * B[34] - B[3] * B[11] * B[14] * B[28] * B[31] - B[3] * B[11] * B[16] * B[25] * B[32] + B[3] * B[11] * B[16] * B[26] * B[31] -
            B[4] * B[7] * B[14] * B[27] * B[35] + B[4] * B[7] * B[14] * B[29] * B[33] + B[4] * B[7] * B[15] * B[26] * B[35] - B[4] * B[7] * B[15] * B[29] * B[32] -
            B[4] * B[7] * B[17] * B[26] * B[33] + B[4] * B[7] * B[17] * B[27] * B[32] + B[4] * B[8] * B[13] * B[27] * B[35] - B[4] * B[8] * B[13] * B[29] * B[33] -
            B[4] * B[8] * B[15] * B[25] * B[35] + B[4] * B[8] * B[15] * B[29] * B[31] + B[4] * B[8] * B[17] * B[25] * B[33] - B[4] * B[8] * B[17] * B[27] * B[31] -
            B[4] * B[9] * B[13] * B[26] * B[35] + B[4] * B[9] * B[13] * B[29] * B[32] + B[4] * B[9] * B[14] * B[25] * B[35] - B[4] * B[9] * B[14] * B[29] * B[31] -
            B[4] * B[9] * B[17] * B[25] * B[32] + B[4] * B[9] * B[17] * B[26] * B[31] + B[4] * B[11] * B[13] * B[26] * B[33] - B[4] * B[11] * B[13] * B[27] * B[32] -
            B[4] * B[11] * B[14] * B[25] * B[33] + B[4] * B[11] * B[14] * B[27] * B[31] + B[4] * B[11] * B[15] * B[25] * B[32] - B[4] * B[11] * B[15] * B[26] * B[31] +
            B[5] * B[7] * B[14] * B[27] * B[34] - B[5] * B[7] * B[14] * B[28] * B[33] - B[5] * B[7] * B[15] * B[26] * B[34] + B[5] * B[7] * B[15] * B[28] * B[32] +
            B[5] * B[7] * B[16] * B[26] * B[33] - B[5] * B[7] * B[16] * B[27] * B[32] - B[5] * B[8] * B[13] * B[27] * B[34] + B[5] * B[8] * B[13] * B[28] * B[33] +
            B[5] * B[8] * B[15] * B[25] * B[34] - B[5] * B[8] * B[15] * B[28] * B[31] - B[5] * B[8] * B[16] * B[25] * B[33] + B[5] * B[8] * B[16] * B[27] * B[31] +
            B[5] * B[9] * B[13] * B[26] * B[34] - B[5] * B[9] * B[13] * B[28] * B[32] - B[5] * B[9] * B[14] * B[25] * B[34] + B[5] * B[9] * B[14] * B[28] * B[31] +
            B[5] * B[9] * B[16] * B[25] * B[32] - B[5] * B[9] * B[16] * B[26] * B[31] - B[5] * B[10] * B[13] * B[26] * B[33] + B[5] * B[10] * B[13] * B[27] * B[32] +
            B[5] * B[10] * B[14] * B[25] * B[33] - B[5] * B[10] * B[14] * B[27] * B[31] - B[5] * B[10] * B[15] * B[25] * B[32] + B[5] * B[10] * B[15] * B[26] * B[31] -
            B[7] * B[14] * B[21] * B[28] * B[35] + B[7] * B[14] * B[21] * B[29] * B[34] + B[7] * B[14] * B[22] * B[27] * B[35] - B[7] * B[14] * B[22] * B[29] * B[33] -
            B[7] * B[14] * B[23] * B[27] * B[34] + B[7] * B[14] * B[23] * B[28] * B[33] + B[7] * B[15] * B[20] * B[28] * B[35] - B[7] * B[15] * B[20] * B[29] * B[34] -
            B[7] * B[15] * B[22] * B[26] * B[35] + B[7] * B[15] * B[22] * B[29] * B[32] + B[7] * B[15] * B[23] * B[26] * B[34] - B[7] * B[15] * B[23] * B[28] * B[32] -
            B[7] * B[16] * B[20] * B[27] * B[35] + B[7] * B[16] * B[20] * B[29] * B[33] + B[7] * B[16] * B[21] * B[26] * B[35] - B[7] * B[16] * B[21] * B[29] * B[32] -
            B[7] * B[16] * B[23] * B[26] * B[33] + B[7] * B[16] * B[23] * B[27] * B[32] + B[7] * B[17] * B[20] * B[27] * B[34] - B[7] * B[17] * B[20] * B[28] * B[33] -
            B[7] * B[17] * B[21] * B[26] * B[34] + B[7] * B[17] * B[21] * B[28] * B[32] + B[7] * B[17] * B[22] * B[26] * B[33] - B[7] * B[17] * B[22] * B[27] * B[32] +
            B[8] * B[13] * B[21] * B[28] * B[35] - B[8] * B[13] * B[21] * B[29] * B[34] - B[8] * B[13] * B[22] * B[27] * B[35] + B[8] * B[13] * B[22] * B[29] * B[33] +
            B[8] * B[13] * B[23] * B[27] * B[34] - B[8] * B[13] * B[23] * B[28] * B[33] - B[8] * B[15] * B[19] * B[28] * B[35] + B[8] * B[15] * B[19] * B[29] * B[34] +
            B[8] * B[15] * B[22] * B[25] * B[35] - B[8] * B[15] * B[22] * B[29] * B[31] - B[8] * B[15] * B[23] * B[25] * B[34] + B[8] * B[15] * B[23] * B[28] * B[31] +
            B[8] * B[16] * B[19] * B[27] * B[35] - B[8] * B[16] * B[19] * B[29] * B[33] - B[8] * B[16] * B[21] * B[25] * B[35] + B[8] * B[16] * B[21] * B[29] * B[31] +
            B[8] * B[16] * B[23] * B[25] * B[33] - B[8] * B[16] * B[23] * B[27] * B[31] - B[8] * B[17] * B[19] * B[27] * B[34] + B[8] * B[17] * B[19] * B[28] * B[33] +
            B[8] * B[17] * B[21] * B[25] * B[34] - B[8] * B[17] * B[21] * B[28] * B[31] - B[8] * B[17] * B[22] * B[25] * B[33] + B[8] * B[17] * B[22] * B[27] * B[31] -
            B[9] * B[13] * B[20] * B[28] * B[35] + B[9] * B[13] * B[20] * B[29] * B[34] + B[9] * B[13] * B[22] * B[26] * B[35] - B[9] * B[13] * B[22] * B[29] * B[32] -
            B[9] * B[13] * B[23] * B[26] * B[34] + B[9] * B[13] * B[23] * B[28] * B[32] + B[9] * B[14] * B[19] * B[28] * B[35] - B[9] * B[14] * B[19] * B[29] * B[34] -
            B[9] * B[14] * B[22] * B[25] * B[35] + B[9] * B[14] * B[22] * B[29] * B[31] + B[9] * B[14] * B[23] * B[25] * B[34] - B[9] * B[14] * B[23] * B[28] * B[31] -
            B[9] * B[16] * B[19] * B[26] * B[35] + B[9] * B[16] * B[19] * B[29] * B[32] + B[9] * B[16] * B[20] * B[25] * B[35] - B[9] * B[16] * B[20] * B[29] * B[31] -
            B[9] * B[16] * B[23] * B[25] * B[32] + B[9] * B[16] * B[23] * B[26] * B[31] + B[9] * B[17] * B[19] * B[26] * B[34] - B[9] * B[17] * B[19] * B[28] * B[32] -
            B[9] * B[17] * B[20] * B[25] * B[34] + B[9] * B[17] * B[20] * B[28] * B[31] + B[9] * B[17] * B[22] * B[25] * B[32] - B[9] * B[17] * B[22] * B[26] * B[31] +
            B[10] * B[13] * B[20] * B[27] * B[35] - B[10] * B[13] * B[20] * B[29] * B[33] - B[10] * B[13] * B[21] * B[26] * B[35] + B[10] * B[13] * B[21] * B[29] * B[32] +
            B[10] * B[13] * B[23] * B[26] * B[33] - B[10] * B[13] * B[23] * B[27] * B[32] - B[10] * B[14] * B[19] * B[27] * B[35] + B[10] * B[14] * B[19] * B[29] * B[33] +
            B[10] * B[14] * B[21] * B[25] * B[35] - B[10] * B[14] * B[21] * B[29] * B[31] - B[10] * B[14] * B[23] * B[25] * B[33] + B[10] * B[14] * B[23] * B[27] * B[31] +
            B[10] * B[15] * B[19] * B[26] * B[35] - B[10] * B[15] * B[19] * B[29] * B[32] - B[10] * B[15] * B[20] * B[25] * B[35] + B[10] * B[15] * B[20] * B[29] * B[31] +
            B[10] * B[15] * B[23] * B[25] * B[32] - B[10] * B[15] * B[23] * B[26] * B[31] - B[10] * B[17] * B[19] * B[26] * B[33] + B[10] * B[17] * B[19] * B[27] * B[32] +
            B[10] * B[17] * B[20] * B[25] * B[33] - B[10] * B[17] * B[20] * B[27] * B[31] - B[10] * B[17] * B[21] * B[25] * B[32] + B[10] * B[17] * B[21] * B[26] * B[31] -
            B[11] * B[13] * B[20] * B[27] * B[34] + B[11] * B[13] * B[20] * B[28] * B[33] + B[11] * B[13] * B[21] * B[26] * B[34] - B[11] * B[13] * B[21] * B[28] * B[32] -
            B[11] * B[13] * B[22] * B[26] * B[33] + B[11] * B[13] * B[22] * B[27] * B[32] + B[11] * B[14] * B[19] * B[27] * B[34] - B[11] * B[14] * B[19] * B[28] * B[33] -
            B[11] * B[14] * B[21] * B[25] * B[34] + B[11] * B[14] * B[21] * B[28] * B[31] + B[11] * B[14] * B[22] * B[25] * B[33] - B[11] * B[14] * B[22] * B[27] * B[31] -
            B[11] * B[15] * B[19] * B[26] * B[34] + B[11] * B[15] * B[19] * B[28] * B[32] + B[11] * B[15] * B[20] * B[25] * B[34] - B[11] * B[15] * B[20] * B[28] * B[31] -
            B[11] * B[15] * B[22] * B[25] * B[32] + B[11] * B[15] * B[22] * B[26] * B[31] + B[11] * B[16] * B[19] * B[26] * B[33] - B[11] * B[16] * B[19] * B[27] * B[32] -
            B[11] * B[16] * B[20] * B[25] * B[33] + B[11] * B[16] * B[20] * B[27] * B[31] + B[11] * B[16] * B[21] * B[25] * B[32] - B[11] * B[16] * B[21] * B[26] * B[31]) /
           (DET);

    L[1] = -(B[0] * B[8] * B[15] * B[22] * B[29] - B[0] * B[8] * B[15] * B[23] * B[28] - B[0] * B[8] * B[16] * B[21] * B[29] + B[0] * B[8] * B[16] * B[23] * B[27] +
             B[0] * B[8] * B[17] * B[21] * B[28] - B[0] * B[8] * B[17] * B[22] * B[27] - B[0] * B[9] * B[14] * B[22] * B[29] + B[0] * B[9] * B[14] * B[23] * B[28] +
             B[0] * B[9] * B[16] * B[20] * B[29] - B[0] * B[9] * B[16] * B[23] * B[26] - B[0] * B[9] * B[17] * B[20] * B[28] + B[0] * B[9] * B[17] * B[22] * B[26] +
             B[0] * B[10] * B[14] * B[21] * B[29] - B[0] * B[10] * B[14] * B[23] * B[27] - B[0] * B[10] * B[15] * B[20] * B[29] + B[0] * B[10] * B[15] * B[23] * B[26] +
             B[0] * B[10] * B[17] * B[20] * B[27] - B[0] * B[10] * B[17] * B[21] * B[26] - B[0] * B[11] * B[14] * B[21] * B[28] + B[0] * B[11] * B[14] * B[22] * B[27] +
             B[0] * B[11] * B[15] * B[20] * B[28] - B[0] * B[11] * B[15] * B[22] * B[26] - B[0] * B[11] * B[16] * B[20] * B[27] + B[0] * B[11] * B[16] * B[21] * B[26] -
             B[2] * B[6] * B[15] * B[22] * B[29] + B[2] * B[6] * B[15] * B[23] * B[28] + B[2] * B[6] * B[16] * B[21] * B[29] - B[2] * B[6] * B[16] * B[23] * B[27] -
             B[2] * B[6] * B[17] * B[21] * B[28] + B[2] * B[6] * B[17] * B[22] * B[27] + B[2] * B[9] * B[12] * B[22] * B[29] - B[2] * B[9] * B[12] * B[23] * B[28] -
             B[2] * B[9] * B[16] * B[18] * B[29] + B[2] * B[9] * B[16] * B[23] * B[24] + B[2] * B[9] * B[17] * B[18] * B[28] - B[2] * B[9] * B[17] * B[22] * B[24] -
             B[2] * B[10] * B[12] * B[21] * B[29] + B[2] * B[10] * B[12] * B[23] * B[27] + B[2] * B[10] * B[15] * B[18] * B[29] - B[2] * B[10] * B[15] * B[23] * B[24] -
             B[2] * B[10] * B[17] * B[18] * B[27] + B[2] * B[10] * B[17] * B[21] * B[24] + B[2] * B[11] * B[12] * B[21] * B[28] - B[2] * B[11] * B[12] * B[22] * B[27] -
             B[2] * B[11] * B[15] * B[18] * B[28] + B[2] * B[11] * B[15] * B[22] * B[24] + B[2] * B[11] * B[16] * B[18] * B[27] - B[2] * B[11] * B[16] * B[21] * B[24] +
             B[3] * B[6] * B[14] * B[22] * B[29] - B[3] * B[6] * B[14] * B[23] * B[28] - B[3] * B[6] * B[16] * B[20] * B[29] + B[3] * B[6] * B[16] * B[23] * B[26] +
             B[3] * B[6] * B[17] * B[20] * B[28] - B[3] * B[6] * B[17] * B[22] * B[26] - B[3] * B[8] * B[12] * B[22] * B[29] + B[3] * B[8] * B[12] * B[23] * B[28] +
             B[3] * B[8] * B[16] * B[18] * B[29] - B[3] * B[8] * B[16] * B[23] * B[24] - B[3] * B[8] * B[17] * B[18] * B[28] + B[3] * B[8] * B[17] * B[22] * B[24] +
             B[3] * B[10] * B[12] * B[20] * B[29] - B[3] * B[10] * B[12] * B[23] * B[26] - B[3] * B[10] * B[14] * B[18] * B[29] + B[3] * B[10] * B[14] * B[23] * B[24] +
             B[3] * B[10] * B[17] * B[18] * B[26] - B[3] * B[10] * B[17] * B[20] * B[24] - B[3] * B[11] * B[12] * B[20] * B[28] + B[3] * B[11] * B[12] * B[22] * B[26] +
             B[3] * B[11] * B[14] * B[18] * B[28] - B[3] * B[11] * B[14] * B[22] * B[24] - B[3] * B[11] * B[16] * B[18] * B[26] + B[3] * B[11] * B[16] * B[20] * B[24] -
             B[4] * B[6] * B[14] * B[21] * B[29] + B[4] * B[6] * B[14] * B[23] * B[27] + B[4] * B[6] * B[15] * B[20] * B[29] - B[4] * B[6] * B[15] * B[23] * B[26] -
             B[4] * B[6] * B[17] * B[20] * B[27] + B[4] * B[6] * B[17] * B[21] * B[26] + B[4] * B[8] * B[12] * B[21] * B[29] - B[4] * B[8] * B[12] * B[23] * B[27] -
             B[4] * B[8] * B[15] * B[18] * B[29] + B[4] * B[8] * B[15] * B[23] * B[24] + B[4] * B[8] * B[17] * B[18] * B[27] - B[4] * B[8] * B[17] * B[21] * B[24] -
             B[4] * B[9] * B[12] * B[20] * B[29] + B[4] * B[9] * B[12] * B[23] * B[26] + B[4] * B[9] * B[14] * B[18] * B[29] - B[4] * B[9] * B[14] * B[23] * B[24] -
             B[4] * B[9] * B[17] * B[18] * B[26] + B[4] * B[9] * B[17] * B[20] * B[24] + B[4] * B[11] * B[12] * B[20] * B[27] - B[4] * B[11] * B[12] * B[21] * B[26] -
             B[4] * B[11] * B[14] * B[18] * B[27] + B[4] * B[11] * B[14] * B[21] * B[24] + B[4] * B[11] * B[15] * B[18] * B[26] - B[4] * B[11] * B[15] * B[20] * B[24] +
             B[5] * B[6] * B[14] * B[21] * B[28] - B[5] * B[6] * B[14] * B[22] * B[27] - B[5] * B[6] * B[15] * B[20] * B[28] + B[5] * B[6] * B[15] * B[22] * B[26] +
             B[5] * B[6] * B[16] * B[20] * B[27] - B[5] * B[6] * B[16] * B[21] * B[26] - B[5] * B[8] * B[12] * B[21] * B[28] + B[5] * B[8] * B[12] * B[22] * B[27] +
             B[5] * B[8] * B[15] * B[18] * B[28] - B[5] * B[8] * B[15] * B[22] * B[24] - B[5] * B[8] * B[16] * B[18] * B[27] + B[5] * B[8] * B[16] * B[21] * B[24] +
             B[5] * B[9] * B[12] * B[20] * B[28] - B[5] * B[9] * B[12] * B[22] * B[26] - B[5] * B[9] * B[14] * B[18] * B[28] + B[5] * B[9] * B[14] * B[22] * B[24] +
             B[5] * B[9] * B[16] * B[18] * B[26] - B[5] * B[9] * B[16] * B[20] * B[24] - B[5] * B[10] * B[12] * B[20] * B[27] + B[5] * B[10] * B[12] * B[21] * B[26] +
             B[5] * B[10] * B[14] * B[18] * B[27] - B[5] * B[10] * B[14] * B[21] * B[24] - B[5] * B[10] * B[15] * B[18] * B[26] + B[5] * B[10] * B[15] * B[20] * B[24] +
             B[0] * B[8] * B[15] * B[28] * B[35] - B[0] * B[8] * B[15] * B[29] * B[34] - B[0] * B[8] * B[16] * B[27] * B[35] + B[0] * B[8] * B[16] * B[29] * B[33] +
             B[0] * B[8] * B[17] * B[27] * B[34] - B[0] * B[8] * B[17] * B[28] * B[33] - B[0] * B[9] * B[14] * B[28] * B[35] + B[0] * B[9] * B[14] * B[29] * B[34] +
             B[0] * B[9] * B[16] * B[26] * B[35] - B[0] * B[9] * B[16] * B[29] * B[32] - B[0] * B[9] * B[17] * B[26] * B[34] + B[0] * B[9] * B[17] * B[28] * B[32] +
             B[0] * B[10] * B[14] * B[27] * B[35] - B[0] * B[10] * B[14] * B[29] * B[33] - B[0] * B[10] * B[15] * B[26] * B[35] + B[0] * B[10] * B[15] * B[29] * B[32] +
             B[0] * B[10] * B[17] * B[26] * B[33] - B[0] * B[10] * B[17] * B[27] * B[32] - B[0] * B[11] * B[14] * B[27] * B[34] + B[0] * B[11] * B[14] * B[28] * B[33] +
             B[0] * B[11] * B[15] * B[26] * B[34] - B[0] * B[11] * B[15] * B[28] * B[32] - B[0] * B[11] * B[16] * B[26] * B[33] + B[0] * B[11] * B[16] * B[27] * B[32] -
             B[2] * B[6] * B[15] * B[28] * B[35] + B[2] * B[6] * B[15] * B[29] * B[34] + B[2] * B[6] * B[16] * B[27] * B[35] - B[2] * B[6] * B[16] * B[29] * B[33] -
             B[2] * B[6] * B[17] * B[27] * B[34] + B[2] * B[6] * B[17] * B[28] * B[33] + B[2] * B[9] * B[12] * B[28] * B[35] - B[2] * B[9] * B[12] * B[29] * B[34] -
             B[2] * B[9] * B[16] * B[24] * B[35] + B[2] * B[9] * B[16] * B[29] * B[30] + B[2] * B[9] * B[17] * B[24] * B[34] - B[2] * B[9] * B[17] * B[28] * B[30] -
             B[2] * B[10] * B[12] * B[27] * B[35] + B[2] * B[10] * B[12] * B[29] * B[33] + B[2] * B[10] * B[15] * B[24] * B[35] - B[2] * B[10] * B[15] * B[29] * B[30] -
             B[2] * B[10] * B[17] * B[24] * B[33] + B[2] * B[10] * B[17] * B[27] * B[30] + B[2] * B[11] * B[12] * B[27] * B[34] - B[2] * B[11] * B[12] * B[28] * B[33] -
             B[2] * B[11] * B[15] * B[24] * B[34] + B[2] * B[11] * B[15] * B[28] * B[30] + B[2] * B[11] * B[16] * B[24] * B[33] - B[2] * B[11] * B[16] * B[27] * B[30] +
             B[3] * B[6] * B[14] * B[28] * B[35] - B[3] * B[6] * B[14] * B[29] * B[34] - B[3] * B[6] * B[16] * B[26] * B[35] + B[3] * B[6] * B[16] * B[29] * B[32] +
             B[3] * B[6] * B[17] * B[26] * B[34] - B[3] * B[6] * B[17] * B[28] * B[32] - B[3] * B[8] * B[12] * B[28] * B[35] + B[3] * B[8] * B[12] * B[29] * B[34] +
             B[3] * B[8] * B[16] * B[24] * B[35] - B[3] * B[8] * B[16] * B[29] * B[30] - B[3] * B[8] * B[17] * B[24] * B[34] + B[3] * B[8] * B[17] * B[28] * B[30] +
             B[3] * B[10] * B[12] * B[26] * B[35] - B[3] * B[10] * B[12] * B[29] * B[32] - B[3] * B[10] * B[14] * B[24] * B[35] + B[3] * B[10] * B[14] * B[29] * B[30] +
             B[3] * B[10] * B[17] * B[24] * B[32] - B[3] * B[10] * B[17] * B[26] * B[30] - B[3] * B[11] * B[12] * B[26] * B[34] + B[3] * B[11] * B[12] * B[28] * B[32] +
             B[3] * B[11] * B[14] * B[24] * B[34] - B[3] * B[11] * B[14] * B[28] * B[30] - B[3] * B[11] * B[16] * B[24] * B[32] + B[3] * B[11] * B[16] * B[26] * B[30] -
             B[4] * B[6] * B[14] * B[27] * B[35] + B[4] * B[6] * B[14] * B[29] * B[33] + B[4] * B[6] * B[15] * B[26] * B[35] - B[4] * B[6] * B[15] * B[29] * B[32] -
             B[4] * B[6] * B[17] * B[26] * B[33] + B[4] * B[6] * B[17] * B[27] * B[32] + B[4] * B[8] * B[12] * B[27] * B[35] - B[4] * B[8] * B[12] * B[29] * B[33] -
             B[4] * B[8] * B[15] * B[24] * B[35] + B[4] * B[8] * B[15] * B[29] * B[30] + B[4] * B[8] * B[17] * B[24] * B[33] - B[4] * B[8] * B[17] * B[27] * B[30] -
             B[4] * B[9] * B[12] * B[26] * B[35] + B[4] * B[9] * B[12] * B[29] * B[32] + B[4] * B[9] * B[14] * B[24] * B[35] - B[4] * B[9] * B[14] * B[29] * B[30] -
             B[4] * B[9] * B[17] * B[24] * B[32] + B[4] * B[9] * B[17] * B[26] * B[30] + B[4] * B[11] * B[12] * B[26] * B[33] - B[4] * B[11] * B[12] * B[27] * B[32] -
             B[4] * B[11] * B[14] * B[24] * B[33] + B[4] * B[11] * B[14] * B[27] * B[30] + B[4] * B[11] * B[15] * B[24] * B[32] - B[4] * B[11] * B[15] * B[26] * B[30] +
             B[5] * B[6] * B[14] * B[27] * B[34] - B[5] * B[6] * B[14] * B[28] * B[33] - B[5] * B[6] * B[15] * B[26] * B[34] + B[5] * B[6] * B[15] * B[28] * B[32] +
             B[5] * B[6] * B[16] * B[26] * B[33] - B[5] * B[6] * B[16] * B[27] * B[32] - B[5] * B[8] * B[12] * B[27] * B[34] + B[5] * B[8] * B[12] * B[28] * B[33] +
             B[5] * B[8] * B[15] * B[24] * B[34] - B[5] * B[8] * B[15] * B[28] * B[30] - B[5] * B[8] * B[16] * B[24] * B[33] + B[5] * B[8] * B[16] * B[27] * B[30] +
             B[5] * B[9] * B[12] * B[26] * B[34] - B[5] * B[9] * B[12] * B[28] * B[32] - B[5] * B[9] * B[14] * B[24] * B[34] + B[5] * B[9] * B[14] * B[28] * B[30] +
             B[5] * B[9] * B[16] * B[24] * B[32] - B[5] * B[9] * B[16] * B[26] * B[30] - B[5] * B[10] * B[12] * B[26] * B[33] + B[5] * B[10] * B[12] * B[27] * B[32] +
             B[5] * B[10] * B[14] * B[24] * B[33] - B[5] * B[10] * B[14] * B[27] * B[30] - B[5] * B[10] * B[15] * B[24] * B[32] + B[5] * B[10] * B[15] * B[26] * B[30] -
             B[6] * B[14] * B[21] * B[28] * B[35] + B[6] * B[14] * B[21] * B[29] * B[34] + B[6] * B[14] * B[22] * B[27] * B[35] - B[6] * B[14] * B[22] * B[29] * B[33] -
             B[6] * B[14] * B[23] * B[27] * B[34] + B[6] * B[14] * B[23] * B[28] * B[33] + B[6] * B[15] * B[20] * B[28] * B[35] - B[6] * B[15] * B[20] * B[29] * B[34] -
             B[6] * B[15] * B[22] * B[26] * B[35] + B[6] * B[15] * B[22] * B[29] * B[32] + B[6] * B[15] * B[23] * B[26] * B[34] - B[6] * B[15] * B[23] * B[28] * B[32] -
             B[6] * B[16] * B[20] * B[27] * B[35] + B[6] * B[16] * B[20] * B[29] * B[33] + B[6] * B[16] * B[21] * B[26] * B[35] - B[6] * B[16] * B[21] * B[29] * B[32] -
             B[6] * B[16] * B[23] * B[26] * B[33] + B[6] * B[16] * B[23] * B[27] * B[32] + B[6] * B[17] * B[20] * B[27] * B[34] - B[6] * B[17] * B[20] * B[28] * B[33] -
             B[6] * B[17] * B[21] * B[26] * B[34] + B[6] * B[17] * B[21] * B[28] * B[32] + B[6] * B[17] * B[22] * B[26] * B[33] - B[6] * B[17] * B[22] * B[27] * B[32] +
             B[8] * B[12] * B[21] * B[28] * B[35] - B[8] * B[12] * B[21] * B[29] * B[34] - B[8] * B[12] * B[22] * B[27] * B[35] + B[8] * B[12] * B[22] * B[29] * B[33] +
             B[8] * B[12] * B[23] * B[27] * B[34] - B[8] * B[12] * B[23] * B[28] * B[33] - B[8] * B[15] * B[18] * B[28] * B[35] + B[8] * B[15] * B[18] * B[29] * B[34] +
             B[8] * B[15] * B[22] * B[24] * B[35] - B[8] * B[15] * B[22] * B[29] * B[30] - B[8] * B[15] * B[23] * B[24] * B[34] + B[8] * B[15] * B[23] * B[28] * B[30] +
             B[8] * B[16] * B[18] * B[27] * B[35] - B[8] * B[16] * B[18] * B[29] * B[33] - B[8] * B[16] * B[21] * B[24] * B[35] + B[8] * B[16] * B[21] * B[29] * B[30] +
             B[8] * B[16] * B[23] * B[24] * B[33] - B[8] * B[16] * B[23] * B[27] * B[30] - B[8] * B[17] * B[18] * B[27] * B[34] + B[8] * B[17] * B[18] * B[28] * B[33] +
             B[8] * B[17] * B[21] * B[24] * B[34] - B[8] * B[17] * B[21] * B[28] * B[30] - B[8] * B[17] * B[22] * B[24] * B[33] + B[8] * B[17] * B[22] * B[27] * B[30] -
             B[9] * B[12] * B[20] * B[28] * B[35] + B[9] * B[12] * B[20] * B[29] * B[34] + B[9] * B[12] * B[22] * B[26] * B[35] - B[9] * B[12] * B[22] * B[29] * B[32] -
             B[9] * B[12] * B[23] * B[26] * B[34] + B[9] * B[12] * B[23] * B[28] * B[32] + B[9] * B[14] * B[18] * B[28] * B[35] - B[9] * B[14] * B[18] * B[29] * B[34] -
             B[9] * B[14] * B[22] * B[24] * B[35] + B[9] * B[14] * B[22] * B[29] * B[30] + B[9] * B[14] * B[23] * B[24] * B[34] - B[9] * B[14] * B[23] * B[28] * B[30] -
             B[9] * B[16] * B[18] * B[26] * B[35] + B[9] * B[16] * B[18] * B[29] * B[32] + B[9] * B[16] * B[20] * B[24] * B[35] - B[9] * B[16] * B[20] * B[29] * B[30] -
             B[9] * B[16] * B[23] * B[24] * B[32] + B[9] * B[16] * B[23] * B[26] * B[30] + B[9] * B[17] * B[18] * B[26] * B[34] - B[9] * B[17] * B[18] * B[28] * B[32] -
             B[9] * B[17] * B[20] * B[24] * B[34] + B[9] * B[17] * B[20] * B[28] * B[30] + B[9] * B[17] * B[22] * B[24] * B[32] - B[9] * B[17] * B[22] * B[26] * B[30] +
             B[10] * B[12] * B[20] * B[27] * B[35] - B[10] * B[12] * B[20] * B[29] * B[33] - B[10] * B[12] * B[21] * B[26] * B[35] + B[10] * B[12] * B[21] * B[29] * B[32] +
             B[10] * B[12] * B[23] * B[26] * B[33] - B[10] * B[12] * B[23] * B[27] * B[32] - B[10] * B[14] * B[18] * B[27] * B[35] + B[10] * B[14] * B[18] * B[29] * B[33] +
             B[10] * B[14] * B[21] * B[24] * B[35] - B[10] * B[14] * B[21] * B[29] * B[30] - B[10] * B[14] * B[23] * B[24] * B[33] + B[10] * B[14] * B[23] * B[27] * B[30] +
             B[10] * B[15] * B[18] * B[26] * B[35] - B[10] * B[15] * B[18] * B[29] * B[32] - B[10] * B[15] * B[20] * B[24] * B[35] + B[10] * B[15] * B[20] * B[29] * B[30] +
             B[10] * B[15] * B[23] * B[24] * B[32] - B[10] * B[15] * B[23] * B[26] * B[30] - B[10] * B[17] * B[18] * B[26] * B[33] + B[10] * B[17] * B[18] * B[27] * B[32] +
             B[10] * B[17] * B[20] * B[24] * B[33] - B[10] * B[17] * B[20] * B[27] * B[30] - B[10] * B[17] * B[21] * B[24] * B[32] + B[10] * B[17] * B[21] * B[26] * B[30] -
             B[11] * B[12] * B[20] * B[27] * B[34] + B[11] * B[12] * B[20] * B[28] * B[33] + B[11] * B[12] * B[21] * B[26] * B[34] - B[11] * B[12] * B[21] * B[28] * B[32] -
             B[11] * B[12] * B[22] * B[26] * B[33] + B[11] * B[12] * B[22] * B[27] * B[32] + B[11] * B[14] * B[18] * B[27] * B[34] - B[11] * B[14] * B[18] * B[28] * B[33] -
             B[11] * B[14] * B[21] * B[24] * B[34] + B[11] * B[14] * B[21] * B[28] * B[30] + B[11] * B[14] * B[22] * B[24] * B[33] - B[11] * B[14] * B[22] * B[27] * B[30] -
             B[11] * B[15] * B[18] * B[26] * B[34] + B[11] * B[15] * B[18] * B[28] * B[32] + B[11] * B[15] * B[20] * B[24] * B[34] - B[11] * B[15] * B[20] * B[28] * B[30] -
             B[11] * B[15] * B[22] * B[24] * B[32] + B[11] * B[15] * B[22] * B[26] * B[30] + B[11] * B[16] * B[18] * B[26] * B[33] - B[11] * B[16] * B[18] * B[27] * B[32] -
             B[11] * B[16] * B[20] * B[24] * B[33] + B[11] * B[16] * B[20] * B[27] * B[30] + B[11] * B[16] * B[21] * B[24] * B[32] - B[11] * B[16] * B[21] * B[26] * B[30]) /
           (DET);

    L[2] = (B[0] * B[7] * B[15] * B[22] * B[29] - B[0] * B[7] * B[15] * B[23] * B[28] - B[0] * B[7] * B[16] * B[21] * B[29] + B[0] * B[7] * B[16] * B[23] * B[27] +
            B[0] * B[7] * B[17] * B[21] * B[28] - B[0] * B[7] * B[17] * B[22] * B[27] - B[0] * B[9] * B[13] * B[22] * B[29] + B[0] * B[9] * B[13] * B[23] * B[28] +
            B[0] * B[9] * B[16] * B[19] * B[29] - B[0] * B[9] * B[16] * B[23] * B[25] - B[0] * B[9] * B[17] * B[19] * B[28] + B[0] * B[9] * B[17] * B[22] * B[25] +
            B[0] * B[10] * B[13] * B[21] * B[29] - B[0] * B[10] * B[13] * B[23] * B[27] - B[0] * B[10] * B[15] * B[19] * B[29] + B[0] * B[10] * B[15] * B[23] * B[25] +
            B[0] * B[10] * B[17] * B[19] * B[27] - B[0] * B[10] * B[17] * B[21] * B[25] - B[0] * B[11] * B[13] * B[21] * B[28] + B[0] * B[11] * B[13] * B[22] * B[27] +
            B[0] * B[11] * B[15] * B[19] * B[28] - B[0] * B[11] * B[15] * B[22] * B[25] - B[0] * B[11] * B[16] * B[19] * B[27] + B[0] * B[11] * B[16] * B[21] * B[25] -
            B[1] * B[6] * B[15] * B[22] * B[29] + B[1] * B[6] * B[15] * B[23] * B[28] + B[1] * B[6] * B[16] * B[21] * B[29] - B[1] * B[6] * B[16] * B[23] * B[27] -
            B[1] * B[6] * B[17] * B[21] * B[28] + B[1] * B[6] * B[17] * B[22] * B[27] + B[1] * B[9] * B[12] * B[22] * B[29] - B[1] * B[9] * B[12] * B[23] * B[28] -
            B[1] * B[9] * B[16] * B[18] * B[29] + B[1] * B[9] * B[16] * B[23] * B[24] + B[1] * B[9] * B[17] * B[18] * B[28] - B[1] * B[9] * B[17] * B[22] * B[24] -
            B[1] * B[10] * B[12] * B[21] * B[29] + B[1] * B[10] * B[12] * B[23] * B[27] + B[1] * B[10] * B[15] * B[18] * B[29] - B[1] * B[10] * B[15] * B[23] * B[24] -
            B[1] * B[10] * B[17] * B[18] * B[27] + B[1] * B[10] * B[17] * B[21] * B[24] + B[1] * B[11] * B[12] * B[21] * B[28] - B[1] * B[11] * B[12] * B[22] * B[27] -
            B[1] * B[11] * B[15] * B[18] * B[28] + B[1] * B[11] * B[15] * B[22] * B[24] + B[1] * B[11] * B[16] * B[18] * B[27] - B[1] * B[11] * B[16] * B[21] * B[24] +
            B[3] * B[6] * B[13] * B[22] * B[29] - B[3] * B[6] * B[13] * B[23] * B[28] - B[3] * B[6] * B[16] * B[19] * B[29] + B[3] * B[6] * B[16] * B[23] * B[25] +
            B[3] * B[6] * B[17] * B[19] * B[28] - B[3] * B[6] * B[17] * B[22] * B[25] - B[3] * B[7] * B[12] * B[22] * B[29] + B[3] * B[7] * B[12] * B[23] * B[28] +
            B[3] * B[7] * B[16] * B[18] * B[29] - B[3] * B[7] * B[16] * B[23] * B[24] - B[3] * B[7] * B[17] * B[18] * B[28] + B[3] * B[7] * B[17] * B[22] * B[24] +
            B[3] * B[10] * B[12] * B[19] * B[29] - B[3] * B[10] * B[12] * B[23] * B[25] - B[3] * B[10] * B[13] * B[18] * B[29] + B[3] * B[10] * B[13] * B[23] * B[24] +
            B[3] * B[10] * B[17] * B[18] * B[25] - B[3] * B[10] * B[17] * B[19] * B[24] - B[3] * B[11] * B[12] * B[19] * B[28] + B[3] * B[11] * B[12] * B[22] * B[25] +
            B[3] * B[11] * B[13] * B[18] * B[28] - B[3] * B[11] * B[13] * B[22] * B[24] - B[3] * B[11] * B[16] * B[18] * B[25] + B[3] * B[11] * B[16] * B[19] * B[24] -
            B[4] * B[6] * B[13] * B[21] * B[29] + B[4] * B[6] * B[13] * B[23] * B[27] + B[4] * B[6] * B[15] * B[19] * B[29] - B[4] * B[6] * B[15] * B[23] * B[25] -
            B[4] * B[6] * B[17] * B[19] * B[27] + B[4] * B[6] * B[17] * B[21] * B[25] + B[4] * B[7] * B[12] * B[21] * B[29] - B[4] * B[7] * B[12] * B[23] * B[27] -
            B[4] * B[7] * B[15] * B[18] * B[29] + B[4] * B[7] * B[15] * B[23] * B[24] + B[4] * B[7] * B[17] * B[18] * B[27] - B[4] * B[7] * B[17] * B[21] * B[24] -
            B[4] * B[9] * B[12] * B[19] * B[29] + B[4] * B[9] * B[12] * B[23] * B[25] + B[4] * B[9] * B[13] * B[18] * B[29] - B[4] * B[9] * B[13] * B[23] * B[24] -
            B[4] * B[9] * B[17] * B[18] * B[25] + B[4] * B[9] * B[17] * B[19] * B[24] + B[4] * B[11] * B[12] * B[19] * B[27] - B[4] * B[11] * B[12] * B[21] * B[25] -
            B[4] * B[11] * B[13] * B[18] * B[27] + B[4] * B[11] * B[13] * B[21] * B[24] + B[4] * B[11] * B[15] * B[18] * B[25] - B[4] * B[11] * B[15] * B[19] * B[24] +
            B[5] * B[6] * B[13] * B[21] * B[28] - B[5] * B[6] * B[13] * B[22] * B[27] - B[5] * B[6] * B[15] * B[19] * B[28] + B[5] * B[6] * B[15] * B[22] * B[25] +
            B[5] * B[6] * B[16] * B[19] * B[27] - B[5] * B[6] * B[16] * B[21] * B[25] - B[5] * B[7] * B[12] * B[21] * B[28] + B[5] * B[7] * B[12] * B[22] * B[27] +
            B[5] * B[7] * B[15] * B[18] * B[28] - B[5] * B[7] * B[15] * B[22] * B[24] - B[5] * B[7] * B[16] * B[18] * B[27] + B[5] * B[7] * B[16] * B[21] * B[24] +
            B[5] * B[9] * B[12] * B[19] * B[28] - B[5] * B[9] * B[12] * B[22] * B[25] - B[5] * B[9] * B[13] * B[18] * B[28] + B[5] * B[9] * B[13] * B[22] * B[24] +
            B[5] * B[9] * B[16] * B[18] * B[25] - B[5] * B[9] * B[16] * B[19] * B[24] - B[5] * B[10] * B[12] * B[19] * B[27] + B[5] * B[10] * B[12] * B[21] * B[25] +
            B[5] * B[10] * B[13] * B[18] * B[27] - B[5] * B[10] * B[13] * B[21] * B[24] - B[5] * B[10] * B[15] * B[18] * B[25] + B[5] * B[10] * B[15] * B[19] * B[24] +
            B[0] * B[7] * B[15] * B[28] * B[35] - B[0] * B[7] * B[15] * B[29] * B[34] - B[0] * B[7] * B[16] * B[27] * B[35] + B[0] * B[7] * B[16] * B[29] * B[33] +
            B[0] * B[7] * B[17] * B[27] * B[34] - B[0] * B[7] * B[17] * B[28] * B[33] - B[0] * B[9] * B[13] * B[28] * B[35] + B[0] * B[9] * B[13] * B[29] * B[34] +
            B[0] * B[9] * B[16] * B[25] * B[35] - B[0] * B[9] * B[16] * B[29] * B[31] - B[0] * B[9] * B[17] * B[25] * B[34] + B[0] * B[9] * B[17] * B[28] * B[31] +
            B[0] * B[10] * B[13] * B[27] * B[35] - B[0] * B[10] * B[13] * B[29] * B[33] - B[0] * B[10] * B[15] * B[25] * B[35] + B[0] * B[10] * B[15] * B[29] * B[31] +
            B[0] * B[10] * B[17] * B[25] * B[33] - B[0] * B[10] * B[17] * B[27] * B[31] - B[0] * B[11] * B[13] * B[27] * B[34] + B[0] * B[11] * B[13] * B[28] * B[33] +
            B[0] * B[11] * B[15] * B[25] * B[34] - B[0] * B[11] * B[15] * B[28] * B[31] - B[0] * B[11] * B[16] * B[25] * B[33] + B[0] * B[11] * B[16] * B[27] * B[31] -
            B[1] * B[6] * B[15] * B[28] * B[35] + B[1] * B[6] * B[15] * B[29] * B[34] + B[1] * B[6] * B[16] * B[27] * B[35] - B[1] * B[6] * B[16] * B[29] * B[33] -
            B[1] * B[6] * B[17] * B[27] * B[34] + B[1] * B[6] * B[17] * B[28] * B[33] + B[1] * B[9] * B[12] * B[28] * B[35] - B[1] * B[9] * B[12] * B[29] * B[34] -
            B[1] * B[9] * B[16] * B[24] * B[35] + B[1] * B[9] * B[16] * B[29] * B[30] + B[1] * B[9] * B[17] * B[24] * B[34] - B[1] * B[9] * B[17] * B[28] * B[30] -
            B[1] * B[10] * B[12] * B[27] * B[35] + B[1] * B[10] * B[12] * B[29] * B[33] + B[1] * B[10] * B[15] * B[24] * B[35] - B[1] * B[10] * B[15] * B[29] * B[30] -
            B[1] * B[10] * B[17] * B[24] * B[33] + B[1] * B[10] * B[17] * B[27] * B[30] + B[1] * B[11] * B[12] * B[27] * B[34] - B[1] * B[11] * B[12] * B[28] * B[33] -
            B[1] * B[11] * B[15] * B[24] * B[34] + B[1] * B[11] * B[15] * B[28] * B[30] + B[1] * B[11] * B[16] * B[24] * B[33] - B[1] * B[11] * B[16] * B[27] * B[30] +
            B[3] * B[6] * B[13] * B[28] * B[35] - B[3] * B[6] * B[13] * B[29] * B[34] - B[3] * B[6] * B[16] * B[25] * B[35] + B[3] * B[6] * B[16] * B[29] * B[31] +
            B[3] * B[6] * B[17] * B[25] * B[34] - B[3] * B[6] * B[17] * B[28] * B[31] - B[3] * B[7] * B[12] * B[28] * B[35] + B[3] * B[7] * B[12] * B[29] * B[34] +
            B[3] * B[7] * B[16] * B[24] * B[35] - B[3] * B[7] * B[16] * B[29] * B[30] - B[3] * B[7] * B[17] * B[24] * B[34] + B[3] * B[7] * B[17] * B[28] * B[30] +
            B[3] * B[10] * B[12] * B[25] * B[35] - B[3] * B[10] * B[12] * B[29] * B[31] - B[3] * B[10] * B[13] * B[24] * B[35] + B[3] * B[10] * B[13] * B[29] * B[30] +
            B[3] * B[10] * B[17] * B[24] * B[31] - B[3] * B[10] * B[17] * B[25] * B[30] - B[3] * B[11] * B[12] * B[25] * B[34] + B[3] * B[11] * B[12] * B[28] * B[31] +
            B[3] * B[11] * B[13] * B[24] * B[34] - B[3] * B[11] * B[13] * B[28] * B[30] - B[3] * B[11] * B[16] * B[24] * B[31] + B[3] * B[11] * B[16] * B[25] * B[30] -
            B[4] * B[6] * B[13] * B[27] * B[35] + B[4] * B[6] * B[13] * B[29] * B[33] + B[4] * B[6] * B[15] * B[25] * B[35] - B[4] * B[6] * B[15] * B[29] * B[31] -
            B[4] * B[6] * B[17] * B[25] * B[33] + B[4] * B[6] * B[17] * B[27] * B[31] + B[4] * B[7] * B[12] * B[27] * B[35] - B[4] * B[7] * B[12] * B[29] * B[33] -
            B[4] * B[7] * B[15] * B[24] * B[35] + B[4] * B[7] * B[15] * B[29] * B[30] + B[4] * B[7] * B[17] * B[24] * B[33] - B[4] * B[7] * B[17] * B[27] * B[30] -
            B[4] * B[9] * B[12] * B[25] * B[35] + B[4] * B[9] * B[12] * B[29] * B[31] + B[4] * B[9] * B[13] * B[24] * B[35] - B[4] * B[9] * B[13] * B[29] * B[30] -
            B[4] * B[9] * B[17] * B[24] * B[31] + B[4] * B[9] * B[17] * B[25] * B[30] + B[4] * B[11] * B[12] * B[25] * B[33] - B[4] * B[11] * B[12] * B[27] * B[31] -
            B[4] * B[11] * B[13] * B[24] * B[33] + B[4] * B[11] * B[13] * B[27] * B[30] + B[4] * B[11] * B[15] * B[24] * B[31] - B[4] * B[11] * B[15] * B[25] * B[30] +
            B[5] * B[6] * B[13] * B[27] * B[34] - B[5] * B[6] * B[13] * B[28] * B[33] - B[5] * B[6] * B[15] * B[25] * B[34] + B[5] * B[6] * B[15] * B[28] * B[31] +
            B[5] * B[6] * B[16] * B[25] * B[33] - B[5] * B[6] * B[16] * B[27] * B[31] - B[5] * B[7] * B[12] * B[27] * B[34] + B[5] * B[7] * B[12] * B[28] * B[33] +
            B[5] * B[7] * B[15] * B[24] * B[34] - B[5] * B[7] * B[15] * B[28] * B[30] - B[5] * B[7] * B[16] * B[24] * B[33] + B[5] * B[7] * B[16] * B[27] * B[30] +
            B[5] * B[9] * B[12] * B[25] * B[34] - B[5] * B[9] * B[12] * B[28] * B[31] - B[5] * B[9] * B[13] * B[24] * B[34] + B[5] * B[9] * B[13] * B[28] * B[30] +
            B[5] * B[9] * B[16] * B[24] * B[31] - B[5] * B[9] * B[16] * B[25] * B[30] - B[5] * B[10] * B[12] * B[25] * B[33] + B[5] * B[10] * B[12] * B[27] * B[31] +
            B[5] * B[10] * B[13] * B[24] * B[33] - B[5] * B[10] * B[13] * B[27] * B[30] - B[5] * B[10] * B[15] * B[24] * B[31] + B[5] * B[10] * B[15] * B[25] * B[30] -
            B[6] * B[13] * B[21] * B[28] * B[35] + B[6] * B[13] * B[21] * B[29] * B[34] + B[6] * B[13] * B[22] * B[27] * B[35] - B[6] * B[13] * B[22] * B[29] * B[33] -
            B[6] * B[13] * B[23] * B[27] * B[34] + B[6] * B[13] * B[23] * B[28] * B[33] + B[6] * B[15] * B[19] * B[28] * B[35] - B[6] * B[15] * B[19] * B[29] * B[34] -
            B[6] * B[15] * B[22] * B[25] * B[35] + B[6] * B[15] * B[22] * B[29] * B[31] + B[6] * B[15] * B[23] * B[25] * B[34] - B[6] * B[15] * B[23] * B[28] * B[31] -
            B[6] * B[16] * B[19] * B[27] * B[35] + B[6] * B[16] * B[19] * B[29] * B[33] + B[6] * B[16] * B[21] * B[25] * B[35] - B[6] * B[16] * B[21] * B[29] * B[31] -
            B[6] * B[16] * B[23] * B[25] * B[33] + B[6] * B[16] * B[23] * B[27] * B[31] + B[6] * B[17] * B[19] * B[27] * B[34] - B[6] * B[17] * B[19] * B[28] * B[33] -
            B[6] * B[17] * B[21] * B[25] * B[34] + B[6] * B[17] * B[21] * B[28] * B[31] + B[6] * B[17] * B[22] * B[25] * B[33] - B[6] * B[17] * B[22] * B[27] * B[31] +
            B[7] * B[12] * B[21] * B[28] * B[35] - B[7] * B[12] * B[21] * B[29] * B[34] - B[7] * B[12] * B[22] * B[27] * B[35] + B[7] * B[12] * B[22] * B[29] * B[33] +
            B[7] * B[12] * B[23] * B[27] * B[34] - B[7] * B[12] * B[23] * B[28] * B[33] - B[7] * B[15] * B[18] * B[28] * B[35] + B[7] * B[15] * B[18] * B[29] * B[34] +
            B[7] * B[15] * B[22] * B[24] * B[35] - B[7] * B[15] * B[22] * B[29] * B[30] - B[7] * B[15] * B[23] * B[24] * B[34] + B[7] * B[15] * B[23] * B[28] * B[30] +
            B[7] * B[16] * B[18] * B[27] * B[35] - B[7] * B[16] * B[18] * B[29] * B[33] - B[7] * B[16] * B[21] * B[24] * B[35] + B[7] * B[16] * B[21] * B[29] * B[30] +
            B[7] * B[16] * B[23] * B[24] * B[33] - B[7] * B[16] * B[23] * B[27] * B[30] - B[7] * B[17] * B[18] * B[27] * B[34] + B[7] * B[17] * B[18] * B[28] * B[33] +
            B[7] * B[17] * B[21] * B[24] * B[34] - B[7] * B[17] * B[21] * B[28] * B[30] - B[7] * B[17] * B[22] * B[24] * B[33] + B[7] * B[17] * B[22] * B[27] * B[30] -
            B[9] * B[12] * B[19] * B[28] * B[35] + B[9] * B[12] * B[19] * B[29] * B[34] + B[9] * B[12] * B[22] * B[25] * B[35] - B[9] * B[12] * B[22] * B[29] * B[31] -
            B[9] * B[12] * B[23] * B[25] * B[34] + B[9] * B[12] * B[23] * B[28] * B[31] + B[9] * B[13] * B[18] * B[28] * B[35] - B[9] * B[13] * B[18] * B[29] * B[34] -
            B[9] * B[13] * B[22] * B[24] * B[35] + B[9] * B[13] * B[22] * B[29] * B[30] + B[9] * B[13] * B[23] * B[24] * B[34] - B[9] * B[13] * B[23] * B[28] * B[30] -
            B[9] * B[16] * B[18] * B[25] * B[35] + B[9] * B[16] * B[18] * B[29] * B[31] + B[9] * B[16] * B[19] * B[24] * B[35] - B[9] * B[16] * B[19] * B[29] * B[30] -
            B[9] * B[16] * B[23] * B[24] * B[31] + B[9] * B[16] * B[23] * B[25] * B[30] + B[9] * B[17] * B[18] * B[25] * B[34] - B[9] * B[17] * B[18] * B[28] * B[31] -
            B[9] * B[17] * B[19] * B[24] * B[34] + B[9] * B[17] * B[19] * B[28] * B[30] + B[9] * B[17] * B[22] * B[24] * B[31] - B[9] * B[17] * B[22] * B[25] * B[30] +
            B[10] * B[12] * B[19] * B[27] * B[35] - B[10] * B[12] * B[19] * B[29] * B[33] - B[10] * B[12] * B[21] * B[25] * B[35] + B[10] * B[12] * B[21] * B[29] * B[31] +
            B[10] * B[12] * B[23] * B[25] * B[33] - B[10] * B[12] * B[23] * B[27] * B[31] - B[10] * B[13] * B[18] * B[27] * B[35] + B[10] * B[13] * B[18] * B[29] * B[33] +
            B[10] * B[13] * B[21] * B[24] * B[35] - B[10] * B[13] * B[21] * B[29] * B[30] - B[10] * B[13] * B[23] * B[24] * B[33] + B[10] * B[13] * B[23] * B[27] * B[30] +
            B[10] * B[15] * B[18] * B[25] * B[35] - B[10] * B[15] * B[18] * B[29] * B[31] - B[10] * B[15] * B[19] * B[24] * B[35] + B[10] * B[15] * B[19] * B[29] * B[30] +
            B[10] * B[15] * B[23] * B[24] * B[31] - B[10] * B[15] * B[23] * B[25] * B[30] - B[10] * B[17] * B[18] * B[25] * B[33] + B[10] * B[17] * B[18] * B[27] * B[31] +
            B[10] * B[17] * B[19] * B[24] * B[33] - B[10] * B[17] * B[19] * B[27] * B[30] - B[10] * B[17] * B[21] * B[24] * B[31] + B[10] * B[17] * B[21] * B[25] * B[30] -
            B[11] * B[12] * B[19] * B[27] * B[34] + B[11] * B[12] * B[19] * B[28] * B[33] + B[11] * B[12] * B[21] * B[25] * B[34] - B[11] * B[12] * B[21] * B[28] * B[31] -
            B[11] * B[12] * B[22] * B[25] * B[33] + B[11] * B[12] * B[22] * B[27] * B[31] + B[11] * B[13] * B[18] * B[27] * B[34] - B[11] * B[13] * B[18] * B[28] * B[33] -
            B[11] * B[13] * B[21] * B[24] * B[34] + B[11] * B[13] * B[21] * B[28] * B[30] + B[11] * B[13] * B[22] * B[24] * B[33] - B[11] * B[13] * B[22] * B[27] * B[30] -
            B[11] * B[15] * B[18] * B[25] * B[34] + B[11] * B[15] * B[18] * B[28] * B[31] + B[11] * B[15] * B[19] * B[24] * B[34] - B[11] * B[15] * B[19] * B[28] * B[30] -
            B[11] * B[15] * B[22] * B[24] * B[31] + B[11] * B[15] * B[22] * B[25] * B[30] + B[11] * B[16] * B[18] * B[25] * B[33] - B[11] * B[16] * B[18] * B[27] * B[31] -
            B[11] * B[16] * B[19] * B[24] * B[33] + B[11] * B[16] * B[19] * B[27] * B[30] + B[11] * B[16] * B[21] * B[24] * B[31] - B[11] * B[16] * B[21] * B[25] * B[30]) /
           (DET);

    L[3] = -(B[0] * B[7] * B[14] * B[22] * B[29] - B[0] * B[7] * B[14] * B[23] * B[28] - B[0] * B[7] * B[16] * B[20] * B[29] + B[0] * B[7] * B[16] * B[23] * B[26] +
             B[0] * B[7] * B[17] * B[20] * B[28] - B[0] * B[7] * B[17] * B[22] * B[26] - B[0] * B[8] * B[13] * B[22] * B[29] + B[0] * B[8] * B[13] * B[23] * B[28] +
             B[0] * B[8] * B[16] * B[19] * B[29] - B[0] * B[8] * B[16] * B[23] * B[25] - B[0] * B[8] * B[17] * B[19] * B[28] + B[0] * B[8] * B[17] * B[22] * B[25] +
             B[0] * B[10] * B[13] * B[20] * B[29] - B[0] * B[10] * B[13] * B[23] * B[26] - B[0] * B[10] * B[14] * B[19] * B[29] + B[0] * B[10] * B[14] * B[23] * B[25] +
             B[0] * B[10] * B[17] * B[19] * B[26] - B[0] * B[10] * B[17] * B[20] * B[25] - B[0] * B[11] * B[13] * B[20] * B[28] + B[0] * B[11] * B[13] * B[22] * B[26] +
             B[0] * B[11] * B[14] * B[19] * B[28] - B[0] * B[11] * B[14] * B[22] * B[25] - B[0] * B[11] * B[16] * B[19] * B[26] + B[0] * B[11] * B[16] * B[20] * B[25] -
             B[1] * B[6] * B[14] * B[22] * B[29] + B[1] * B[6] * B[14] * B[23] * B[28] + B[1] * B[6] * B[16] * B[20] * B[29] - B[1] * B[6] * B[16] * B[23] * B[26] -
             B[1] * B[6] * B[17] * B[20] * B[28] + B[1] * B[6] * B[17] * B[22] * B[26] + B[1] * B[8] * B[12] * B[22] * B[29] - B[1] * B[8] * B[12] * B[23] * B[28] -
             B[1] * B[8] * B[16] * B[18] * B[29] + B[1] * B[8] * B[16] * B[23] * B[24] + B[1] * B[8] * B[17] * B[18] * B[28] - B[1] * B[8] * B[17] * B[22] * B[24] -
             B[1] * B[10] * B[12] * B[20] * B[29] + B[1] * B[10] * B[12] * B[23] * B[26] + B[1] * B[10] * B[14] * B[18] * B[29] - B[1] * B[10] * B[14] * B[23] * B[24] -
             B[1] * B[10] * B[17] * B[18] * B[26] + B[1] * B[10] * B[17] * B[20] * B[24] + B[1] * B[11] * B[12] * B[20] * B[28] - B[1] * B[11] * B[12] * B[22] * B[26] -
             B[1] * B[11] * B[14] * B[18] * B[28] + B[1] * B[11] * B[14] * B[22] * B[24] + B[1] * B[11] * B[16] * B[18] * B[26] - B[1] * B[11] * B[16] * B[20] * B[24] +
             B[2] * B[6] * B[13] * B[22] * B[29] - B[2] * B[6] * B[13] * B[23] * B[28] - B[2] * B[6] * B[16] * B[19] * B[29] + B[2] * B[6] * B[16] * B[23] * B[25] +
             B[2] * B[6] * B[17] * B[19] * B[28] - B[2] * B[6] * B[17] * B[22] * B[25] - B[2] * B[7] * B[12] * B[22] * B[29] + B[2] * B[7] * B[12] * B[23] * B[28] +
             B[2] * B[7] * B[16] * B[18] * B[29] - B[2] * B[7] * B[16] * B[23] * B[24] - B[2] * B[7] * B[17] * B[18] * B[28] + B[2] * B[7] * B[17] * B[22] * B[24] +
             B[2] * B[10] * B[12] * B[19] * B[29] - B[2] * B[10] * B[12] * B[23] * B[25] - B[2] * B[10] * B[13] * B[18] * B[29] + B[2] * B[10] * B[13] * B[23] * B[24] +
             B[2] * B[10] * B[17] * B[18] * B[25] - B[2] * B[10] * B[17] * B[19] * B[24] - B[2] * B[11] * B[12] * B[19] * B[28] + B[2] * B[11] * B[12] * B[22] * B[25] +
             B[2] * B[11] * B[13] * B[18] * B[28] - B[2] * B[11] * B[13] * B[22] * B[24] - B[2] * B[11] * B[16] * B[18] * B[25] + B[2] * B[11] * B[16] * B[19] * B[24] -
             B[4] * B[6] * B[13] * B[20] * B[29] + B[4] * B[6] * B[13] * B[23] * B[26] + B[4] * B[6] * B[14] * B[19] * B[29] - B[4] * B[6] * B[14] * B[23] * B[25] -
             B[4] * B[6] * B[17] * B[19] * B[26] + B[4] * B[6] * B[17] * B[20] * B[25] + B[4] * B[7] * B[12] * B[20] * B[29] - B[4] * B[7] * B[12] * B[23] * B[26] -
             B[4] * B[7] * B[14] * B[18] * B[29] + B[4] * B[7] * B[14] * B[23] * B[24] + B[4] * B[7] * B[17] * B[18] * B[26] - B[4] * B[7] * B[17] * B[20] * B[24] -
             B[4] * B[8] * B[12] * B[19] * B[29] + B[4] * B[8] * B[12] * B[23] * B[25] + B[4] * B[8] * B[13] * B[18] * B[29] - B[4] * B[8] * B[13] * B[23] * B[24] -
             B[4] * B[8] * B[17] * B[18] * B[25] + B[4] * B[8] * B[17] * B[19] * B[24] + B[4] * B[11] * B[12] * B[19] * B[26] - B[4] * B[11] * B[12] * B[20] * B[25] -
             B[4] * B[11] * B[13] * B[18] * B[26] + B[4] * B[11] * B[13] * B[20] * B[24] + B[4] * B[11] * B[14] * B[18] * B[25] - B[4] * B[11] * B[14] * B[19] * B[24] +
             B[5] * B[6] * B[13] * B[20] * B[28] - B[5] * B[6] * B[13] * B[22] * B[26] - B[5] * B[6] * B[14] * B[19] * B[28] + B[5] * B[6] * B[14] * B[22] * B[25] +
             B[5] * B[6] * B[16] * B[19] * B[26] - B[5] * B[6] * B[16] * B[20] * B[25] - B[5] * B[7] * B[12] * B[20] * B[28] + B[5] * B[7] * B[12] * B[22] * B[26] +
             B[5] * B[7] * B[14] * B[18] * B[28] - B[5] * B[7] * B[14] * B[22] * B[24] - B[5] * B[7] * B[16] * B[18] * B[26] + B[5] * B[7] * B[16] * B[20] * B[24] +
             B[5] * B[8] * B[12] * B[19] * B[28] - B[5] * B[8] * B[12] * B[22] * B[25] - B[5] * B[8] * B[13] * B[18] * B[28] + B[5] * B[8] * B[13] * B[22] * B[24] +
             B[5] * B[8] * B[16] * B[18] * B[25] - B[5] * B[8] * B[16] * B[19] * B[24] - B[5] * B[10] * B[12] * B[19] * B[26] + B[5] * B[10] * B[12] * B[20] * B[25] +
             B[5] * B[10] * B[13] * B[18] * B[26] - B[5] * B[10] * B[13] * B[20] * B[24] - B[5] * B[10] * B[14] * B[18] * B[25] + B[5] * B[10] * B[14] * B[19] * B[24] +
             B[0] * B[7] * B[14] * B[28] * B[35] - B[0] * B[7] * B[14] * B[29] * B[34] - B[0] * B[7] * B[16] * B[26] * B[35] + B[0] * B[7] * B[16] * B[29] * B[32] +
             B[0] * B[7] * B[17] * B[26] * B[34] - B[0] * B[7] * B[17] * B[28] * B[32] - B[0] * B[8] * B[13] * B[28] * B[35] + B[0] * B[8] * B[13] * B[29] * B[34] +
             B[0] * B[8] * B[16] * B[25] * B[35] - B[0] * B[8] * B[16] * B[29] * B[31] - B[0] * B[8] * B[17] * B[25] * B[34] + B[0] * B[8] * B[17] * B[28] * B[31] +
             B[0] * B[10] * B[13] * B[26] * B[35] - B[0] * B[10] * B[13] * B[29] * B[32] - B[0] * B[10] * B[14] * B[25] * B[35] + B[0] * B[10] * B[14] * B[29] * B[31] +
             B[0] * B[10] * B[17] * B[25] * B[32] - B[0] * B[10] * B[17] * B[26] * B[31] - B[0] * B[11] * B[13] * B[26] * B[34] + B[0] * B[11] * B[13] * B[28] * B[32] +
             B[0] * B[11] * B[14] * B[25] * B[34] - B[0] * B[11] * B[14] * B[28] * B[31] - B[0] * B[11] * B[16] * B[25] * B[32] + B[0] * B[11] * B[16] * B[26] * B[31] -
             B[1] * B[6] * B[14] * B[28] * B[35] + B[1] * B[6] * B[14] * B[29] * B[34] + B[1] * B[6] * B[16] * B[26] * B[35] - B[1] * B[6] * B[16] * B[29] * B[32] -
             B[1] * B[6] * B[17] * B[26] * B[34] + B[1] * B[6] * B[17] * B[28] * B[32] + B[1] * B[8] * B[12] * B[28] * B[35] - B[1] * B[8] * B[12] * B[29] * B[34] -
             B[1] * B[8] * B[16] * B[24] * B[35] + B[1] * B[8] * B[16] * B[29] * B[30] + B[1] * B[8] * B[17] * B[24] * B[34] - B[1] * B[8] * B[17] * B[28] * B[30] -
             B[1] * B[10] * B[12] * B[26] * B[35] + B[1] * B[10] * B[12] * B[29] * B[32] + B[1] * B[10] * B[14] * B[24] * B[35] - B[1] * B[10] * B[14] * B[29] * B[30] -
             B[1] * B[10] * B[17] * B[24] * B[32] + B[1] * B[10] * B[17] * B[26] * B[30] + B[1] * B[11] * B[12] * B[26] * B[34] - B[1] * B[11] * B[12] * B[28] * B[32] -
             B[1] * B[11] * B[14] * B[24] * B[34] + B[1] * B[11] * B[14] * B[28] * B[30] + B[1] * B[11] * B[16] * B[24] * B[32] - B[1] * B[11] * B[16] * B[26] * B[30] +
             B[2] * B[6] * B[13] * B[28] * B[35] - B[2] * B[6] * B[13] * B[29] * B[34] - B[2] * B[6] * B[16] * B[25] * B[35] + B[2] * B[6] * B[16] * B[29] * B[31] +
             B[2] * B[6] * B[17] * B[25] * B[34] - B[2] * B[6] * B[17] * B[28] * B[31] - B[2] * B[7] * B[12] * B[28] * B[35] + B[2] * B[7] * B[12] * B[29] * B[34] +
             B[2] * B[7] * B[16] * B[24] * B[35] - B[2] * B[7] * B[16] * B[29] * B[30] - B[2] * B[7] * B[17] * B[24] * B[34] + B[2] * B[7] * B[17] * B[28] * B[30] +
             B[2] * B[10] * B[12] * B[25] * B[35] - B[2] * B[10] * B[12] * B[29] * B[31] - B[2] * B[10] * B[13] * B[24] * B[35] + B[2] * B[10] * B[13] * B[29] * B[30] +
             B[2] * B[10] * B[17] * B[24] * B[31] - B[2] * B[10] * B[17] * B[25] * B[30] - B[2] * B[11] * B[12] * B[25] * B[34] + B[2] * B[11] * B[12] * B[28] * B[31] +
             B[2] * B[11] * B[13] * B[24] * B[34] - B[2] * B[11] * B[13] * B[28] * B[30] - B[2] * B[11] * B[16] * B[24] * B[31] + B[2] * B[11] * B[16] * B[25] * B[30] -
             B[4] * B[6] * B[13] * B[26] * B[35] + B[4] * B[6] * B[13] * B[29] * B[32] + B[4] * B[6] * B[14] * B[25] * B[35] - B[4] * B[6] * B[14] * B[29] * B[31] -
             B[4] * B[6] * B[17] * B[25] * B[32] + B[4] * B[6] * B[17] * B[26] * B[31] + B[4] * B[7] * B[12] * B[26] * B[35] - B[4] * B[7] * B[12] * B[29] * B[32] -
             B[4] * B[7] * B[14] * B[24] * B[35] + B[4] * B[7] * B[14] * B[29] * B[30] + B[4] * B[7] * B[17] * B[24] * B[32] - B[4] * B[7] * B[17] * B[26] * B[30] -
             B[4] * B[8] * B[12] * B[25] * B[35] + B[4] * B[8] * B[12] * B[29] * B[31] + B[4] * B[8] * B[13] * B[24] * B[35] - B[4] * B[8] * B[13] * B[29] * B[30] -
             B[4] * B[8] * B[17] * B[24] * B[31] + B[4] * B[8] * B[17] * B[25] * B[30] + B[4] * B[11] * B[12] * B[25] * B[32] - B[4] * B[11] * B[12] * B[26] * B[31] -
             B[4] * B[11] * B[13] * B[24] * B[32] + B[4] * B[11] * B[13] * B[26] * B[30] + B[4] * B[11] * B[14] * B[24] * B[31] - B[4] * B[11] * B[14] * B[25] * B[30] +
             B[5] * B[6] * B[13] * B[26] * B[34] - B[5] * B[6] * B[13] * B[28] * B[32] - B[5] * B[6] * B[14] * B[25] * B[34] + B[5] * B[6] * B[14] * B[28] * B[31] +
             B[5] * B[6] * B[16] * B[25] * B[32] - B[5] * B[6] * B[16] * B[26] * B[31] - B[5] * B[7] * B[12] * B[26] * B[34] + B[5] * B[7] * B[12] * B[28] * B[32] +
             B[5] * B[7] * B[14] * B[24] * B[34] - B[5] * B[7] * B[14] * B[28] * B[30] - B[5] * B[7] * B[16] * B[24] * B[32] + B[5] * B[7] * B[16] * B[26] * B[30] +
             B[5] * B[8] * B[12] * B[25] * B[34] - B[5] * B[8] * B[12] * B[28] * B[31] - B[5] * B[8] * B[13] * B[24] * B[34] + B[5] * B[8] * B[13] * B[28] * B[30] +
             B[5] * B[8] * B[16] * B[24] * B[31] - B[5] * B[8] * B[16] * B[25] * B[30] - B[5] * B[10] * B[12] * B[25] * B[32] + B[5] * B[10] * B[12] * B[26] * B[31] +
             B[5] * B[10] * B[13] * B[24] * B[32] - B[5] * B[10] * B[13] * B[26] * B[30] - B[5] * B[10] * B[14] * B[24] * B[31] + B[5] * B[10] * B[14] * B[25] * B[30] -
             B[6] * B[13] * B[20] * B[28] * B[35] + B[6] * B[13] * B[20] * B[29] * B[34] + B[6] * B[13] * B[22] * B[26] * B[35] - B[6] * B[13] * B[22] * B[29] * B[32] -
             B[6] * B[13] * B[23] * B[26] * B[34] + B[6] * B[13] * B[23] * B[28] * B[32] + B[6] * B[14] * B[19] * B[28] * B[35] - B[6] * B[14] * B[19] * B[29] * B[34] -
             B[6] * B[14] * B[22] * B[25] * B[35] + B[6] * B[14] * B[22] * B[29] * B[31] + B[6] * B[14] * B[23] * B[25] * B[34] - B[6] * B[14] * B[23] * B[28] * B[31] -
             B[6] * B[16] * B[19] * B[26] * B[35] + B[6] * B[16] * B[19] * B[29] * B[32] + B[6] * B[16] * B[20] * B[25] * B[35] - B[6] * B[16] * B[20] * B[29] * B[31] -
             B[6] * B[16] * B[23] * B[25] * B[32] + B[6] * B[16] * B[23] * B[26] * B[31] + B[6] * B[17] * B[19] * B[26] * B[34] - B[6] * B[17] * B[19] * B[28] * B[32] -
             B[6] * B[17] * B[20] * B[25] * B[34] + B[6] * B[17] * B[20] * B[28] * B[31] + B[6] * B[17] * B[22] * B[25] * B[32] - B[6] * B[17] * B[22] * B[26] * B[31] +
             B[7] * B[12] * B[20] * B[28] * B[35] - B[7] * B[12] * B[20] * B[29] * B[34] - B[7] * B[12] * B[22] * B[26] * B[35] + B[7] * B[12] * B[22] * B[29] * B[32] +
             B[7] * B[12] * B[23] * B[26] * B[34] - B[7] * B[12] * B[23] * B[28] * B[32] - B[7] * B[14] * B[18] * B[28] * B[35] + B[7] * B[14] * B[18] * B[29] * B[34] +
             B[7] * B[14] * B[22] * B[24] * B[35] - B[7] * B[14] * B[22] * B[29] * B[30] - B[7] * B[14] * B[23] * B[24] * B[34] + B[7] * B[14] * B[23] * B[28] * B[30] +
             B[7] * B[16] * B[18] * B[26] * B[35] - B[7] * B[16] * B[18] * B[29] * B[32] - B[7] * B[16] * B[20] * B[24] * B[35] + B[7] * B[16] * B[20] * B[29] * B[30] +
             B[7] * B[16] * B[23] * B[24] * B[32] - B[7] * B[16] * B[23] * B[26] * B[30] - B[7] * B[17] * B[18] * B[26] * B[34] + B[7] * B[17] * B[18] * B[28] * B[32] +
             B[7] * B[17] * B[20] * B[24] * B[34] - B[7] * B[17] * B[20] * B[28] * B[30] - B[7] * B[17] * B[22] * B[24] * B[32] + B[7] * B[17] * B[22] * B[26] * B[30] -
             B[8] * B[12] * B[19] * B[28] * B[35] + B[8] * B[12] * B[19] * B[29] * B[34] + B[8] * B[12] * B[22] * B[25] * B[35] - B[8] * B[12] * B[22] * B[29] * B[31] -
             B[8] * B[12] * B[23] * B[25] * B[34] + B[8] * B[12] * B[23] * B[28] * B[31] + B[8] * B[13] * B[18] * B[28] * B[35] - B[8] * B[13] * B[18] * B[29] * B[34] -
             B[8] * B[13] * B[22] * B[24] * B[35] + B[8] * B[13] * B[22] * B[29] * B[30] + B[8] * B[13] * B[23] * B[24] * B[34] - B[8] * B[13] * B[23] * B[28] * B[30] -
             B[8] * B[16] * B[18] * B[25] * B[35] + B[8] * B[16] * B[18] * B[29] * B[31] + B[8] * B[16] * B[19] * B[24] * B[35] - B[8] * B[16] * B[19] * B[29] * B[30] -
             B[8] * B[16] * B[23] * B[24] * B[31] + B[8] * B[16] * B[23] * B[25] * B[30] + B[8] * B[17] * B[18] * B[25] * B[34] - B[8] * B[17] * B[18] * B[28] * B[31] -
             B[8] * B[17] * B[19] * B[24] * B[34] + B[8] * B[17] * B[19] * B[28] * B[30] + B[8] * B[17] * B[22] * B[24] * B[31] - B[8] * B[17] * B[22] * B[25] * B[30] +
             B[10] * B[12] * B[19] * B[26] * B[35] - B[10] * B[12] * B[19] * B[29] * B[32] - B[10] * B[12] * B[20] * B[25] * B[35] + B[10] * B[12] * B[20] * B[29] * B[31] +
             B[10] * B[12] * B[23] * B[25] * B[32] - B[10] * B[12] * B[23] * B[26] * B[31] - B[10] * B[13] * B[18] * B[26] * B[35] + B[10] * B[13] * B[18] * B[29] * B[32] +
             B[10] * B[13] * B[20] * B[24] * B[35] - B[10] * B[13] * B[20] * B[29] * B[30] - B[10] * B[13] * B[23] * B[24] * B[32] + B[10] * B[13] * B[23] * B[26] * B[30] +
             B[10] * B[14] * B[18] * B[25] * B[35] - B[10] * B[14] * B[18] * B[29] * B[31] - B[10] * B[14] * B[19] * B[24] * B[35] + B[10] * B[14] * B[19] * B[29] * B[30] +
             B[10] * B[14] * B[23] * B[24] * B[31] - B[10] * B[14] * B[23] * B[25] * B[30] - B[10] * B[17] * B[18] * B[25] * B[32] + B[10] * B[17] * B[18] * B[26] * B[31] +
             B[10] * B[17] * B[19] * B[24] * B[32] - B[10] * B[17] * B[19] * B[26] * B[30] - B[10] * B[17] * B[20] * B[24] * B[31] + B[10] * B[17] * B[20] * B[25] * B[30] -
             B[11] * B[12] * B[19] * B[26] * B[34] + B[11] * B[12] * B[19] * B[28] * B[32] + B[11] * B[12] * B[20] * B[25] * B[34] - B[11] * B[12] * B[20] * B[28] * B[31] -
             B[11] * B[12] * B[22] * B[25] * B[32] + B[11] * B[12] * B[22] * B[26] * B[31] + B[11] * B[13] * B[18] * B[26] * B[34] - B[11] * B[13] * B[18] * B[28] * B[32] -
             B[11] * B[13] * B[20] * B[24] * B[34] + B[11] * B[13] * B[20] * B[28] * B[30] + B[11] * B[13] * B[22] * B[24] * B[32] - B[11] * B[13] * B[22] * B[26] * B[30] -
             B[11] * B[14] * B[18] * B[25] * B[34] + B[11] * B[14] * B[18] * B[28] * B[31] + B[11] * B[14] * B[19] * B[24] * B[34] - B[11] * B[14] * B[19] * B[28] * B[30] -
             B[11] * B[14] * B[22] * B[24] * B[31] + B[11] * B[14] * B[22] * B[25] * B[30] + B[11] * B[16] * B[18] * B[25] * B[32] - B[11] * B[16] * B[18] * B[26] * B[31] -
             B[11] * B[16] * B[19] * B[24] * B[32] + B[11] * B[16] * B[19] * B[26] * B[30] + B[11] * B[16] * B[20] * B[24] * B[31] - B[11] * B[16] * B[20] * B[25] * B[30]) /
           (DET);

    L[4] = (B[0] * B[7] * B[14] * B[21] * B[29] - B[0] * B[7] * B[14] * B[23] * B[27] - B[0] * B[7] * B[15] * B[20] * B[29] + B[0] * B[7] * B[15] * B[23] * B[26] +
            B[0] * B[7] * B[17] * B[20] * B[27] - B[0] * B[7] * B[17] * B[21] * B[26] - B[0] * B[8] * B[13] * B[21] * B[29] + B[0] * B[8] * B[13] * B[23] * B[27] +
            B[0] * B[8] * B[15] * B[19] * B[29] - B[0] * B[8] * B[15] * B[23] * B[25] - B[0] * B[8] * B[17] * B[19] * B[27] + B[0] * B[8] * B[17] * B[21] * B[25] +
            B[0] * B[9] * B[13] * B[20] * B[29] - B[0] * B[9] * B[13] * B[23] * B[26] - B[0] * B[9] * B[14] * B[19] * B[29] + B[0] * B[9] * B[14] * B[23] * B[25] +
            B[0] * B[9] * B[17] * B[19] * B[26] - B[0] * B[9] * B[17] * B[20] * B[25] - B[0] * B[11] * B[13] * B[20] * B[27] + B[0] * B[11] * B[13] * B[21] * B[26] +
            B[0] * B[11] * B[14] * B[19] * B[27] - B[0] * B[11] * B[14] * B[21] * B[25] - B[0] * B[11] * B[15] * B[19] * B[26] + B[0] * B[11] * B[15] * B[20] * B[25] -
            B[1] * B[6] * B[14] * B[21] * B[29] + B[1] * B[6] * B[14] * B[23] * B[27] + B[1] * B[6] * B[15] * B[20] * B[29] - B[1] * B[6] * B[15] * B[23] * B[26] -
            B[1] * B[6] * B[17] * B[20] * B[27] + B[1] * B[6] * B[17] * B[21] * B[26] + B[1] * B[8] * B[12] * B[21] * B[29] - B[1] * B[8] * B[12] * B[23] * B[27] -
            B[1] * B[8] * B[15] * B[18] * B[29] + B[1] * B[8] * B[15] * B[23] * B[24] + B[1] * B[8] * B[17] * B[18] * B[27] - B[1] * B[8] * B[17] * B[21] * B[24] -
            B[1] * B[9] * B[12] * B[20] * B[29] + B[1] * B[9] * B[12] * B[23] * B[26] + B[1] * B[9] * B[14] * B[18] * B[29] - B[1] * B[9] * B[14] * B[23] * B[24] -
            B[1] * B[9] * B[17] * B[18] * B[26] + B[1] * B[9] * B[17] * B[20] * B[24] + B[1] * B[11] * B[12] * B[20] * B[27] - B[1] * B[11] * B[12] * B[21] * B[26] -
            B[1] * B[11] * B[14] * B[18] * B[27] + B[1] * B[11] * B[14] * B[21] * B[24] + B[1] * B[11] * B[15] * B[18] * B[26] - B[1] * B[11] * B[15] * B[20] * B[24] +
            B[2] * B[6] * B[13] * B[21] * B[29] - B[2] * B[6] * B[13] * B[23] * B[27] - B[2] * B[6] * B[15] * B[19] * B[29] + B[2] * B[6] * B[15] * B[23] * B[25] +
            B[2] * B[6] * B[17] * B[19] * B[27] - B[2] * B[6] * B[17] * B[21] * B[25] - B[2] * B[7] * B[12] * B[21] * B[29] + B[2] * B[7] * B[12] * B[23] * B[27] +
            B[2] * B[7] * B[15] * B[18] * B[29] - B[2] * B[7] * B[15] * B[23] * B[24] - B[2] * B[7] * B[17] * B[18] * B[27] + B[2] * B[7] * B[17] * B[21] * B[24] +
            B[2] * B[9] * B[12] * B[19] * B[29] - B[2] * B[9] * B[12] * B[23] * B[25] - B[2] * B[9] * B[13] * B[18] * B[29] + B[2] * B[9] * B[13] * B[23] * B[24] +
            B[2] * B[9] * B[17] * B[18] * B[25] - B[2] * B[9] * B[17] * B[19] * B[24] - B[2] * B[11] * B[12] * B[19] * B[27] + B[2] * B[11] * B[12] * B[21] * B[25] +
            B[2] * B[11] * B[13] * B[18] * B[27] - B[2] * B[11] * B[13] * B[21] * B[24] - B[2] * B[11] * B[15] * B[18] * B[25] + B[2] * B[11] * B[15] * B[19] * B[24] -
            B[3] * B[6] * B[13] * B[20] * B[29] + B[3] * B[6] * B[13] * B[23] * B[26] + B[3] * B[6] * B[14] * B[19] * B[29] - B[3] * B[6] * B[14] * B[23] * B[25] -
            B[3] * B[6] * B[17] * B[19] * B[26] + B[3] * B[6] * B[17] * B[20] * B[25] + B[3] * B[7] * B[12] * B[20] * B[29] - B[3] * B[7] * B[12] * B[23] * B[26] -
            B[3] * B[7] * B[14] * B[18] * B[29] + B[3] * B[7] * B[14] * B[23] * B[24] + B[3] * B[7] * B[17] * B[18] * B[26] - B[3] * B[7] * B[17] * B[20] * B[24] -
            B[3] * B[8] * B[12] * B[19] * B[29] + B[3] * B[8] * B[12] * B[23] * B[25] + B[3] * B[8] * B[13] * B[18] * B[29] - B[3] * B[8] * B[13] * B[23] * B[24] -
            B[3] * B[8] * B[17] * B[18] * B[25] + B[3] * B[8] * B[17] * B[19] * B[24] + B[3] * B[11] * B[12] * B[19] * B[26] - B[3] * B[11] * B[12] * B[20] * B[25] -
            B[3] * B[11] * B[13] * B[18] * B[26] + B[3] * B[11] * B[13] * B[20] * B[24] + B[3] * B[11] * B[14] * B[18] * B[25] - B[3] * B[11] * B[14] * B[19] * B[24] +
            B[5] * B[6] * B[13] * B[20] * B[27] - B[5] * B[6] * B[13] * B[21] * B[26] - B[5] * B[6] * B[14] * B[19] * B[27] + B[5] * B[6] * B[14] * B[21] * B[25] +
            B[5] * B[6] * B[15] * B[19] * B[26] - B[5] * B[6] * B[15] * B[20] * B[25] - B[5] * B[7] * B[12] * B[20] * B[27] + B[5] * B[7] * B[12] * B[21] * B[26] +
            B[5] * B[7] * B[14] * B[18] * B[27] - B[5] * B[7] * B[14] * B[21] * B[24] - B[5] * B[7] * B[15] * B[18] * B[26] + B[5] * B[7] * B[15] * B[20] * B[24] +
            B[5] * B[8] * B[12] * B[19] * B[27] - B[5] * B[8] * B[12] * B[21] * B[25] - B[5] * B[8] * B[13] * B[18] * B[27] + B[5] * B[8] * B[13] * B[21] * B[24] +
            B[5] * B[8] * B[15] * B[18] * B[25] - B[5] * B[8] * B[15] * B[19] * B[24] - B[5] * B[9] * B[12] * B[19] * B[26] + B[5] * B[9] * B[12] * B[20] * B[25] +
            B[5] * B[9] * B[13] * B[18] * B[26] - B[5] * B[9] * B[13] * B[20] * B[24] - B[5] * B[9] * B[14] * B[18] * B[25] + B[5] * B[9] * B[14] * B[19] * B[24] +
            B[0] * B[7] * B[14] * B[27] * B[35] - B[0] * B[7] * B[14] * B[29] * B[33] - B[0] * B[7] * B[15] * B[26] * B[35] + B[0] * B[7] * B[15] * B[29] * B[32] +
            B[0] * B[7] * B[17] * B[26] * B[33] - B[0] * B[7] * B[17] * B[27] * B[32] - B[0] * B[8] * B[13] * B[27] * B[35] + B[0] * B[8] * B[13] * B[29] * B[33] +
            B[0] * B[8] * B[15] * B[25] * B[35] - B[0] * B[8] * B[15] * B[29] * B[31] - B[0] * B[8] * B[17] * B[25] * B[33] + B[0] * B[8] * B[17] * B[27] * B[31] +
            B[0] * B[9] * B[13] * B[26] * B[35] - B[0] * B[9] * B[13] * B[29] * B[32] - B[0] * B[9] * B[14] * B[25] * B[35] + B[0] * B[9] * B[14] * B[29] * B[31] +
            B[0] * B[9] * B[17] * B[25] * B[32] - B[0] * B[9] * B[17] * B[26] * B[31] - B[0] * B[11] * B[13] * B[26] * B[33] + B[0] * B[11] * B[13] * B[27] * B[32] +
            B[0] * B[11] * B[14] * B[25] * B[33] - B[0] * B[11] * B[14] * B[27] * B[31] - B[0] * B[11] * B[15] * B[25] * B[32] + B[0] * B[11] * B[15] * B[26] * B[31] -
            B[1] * B[6] * B[14] * B[27] * B[35] + B[1] * B[6] * B[14] * B[29] * B[33] + B[1] * B[6] * B[15] * B[26] * B[35] - B[1] * B[6] * B[15] * B[29] * B[32] -
            B[1] * B[6] * B[17] * B[26] * B[33] + B[1] * B[6] * B[17] * B[27] * B[32] + B[1] * B[8] * B[12] * B[27] * B[35] - B[1] * B[8] * B[12] * B[29] * B[33] -
            B[1] * B[8] * B[15] * B[24] * B[35] + B[1] * B[8] * B[15] * B[29] * B[30] + B[1] * B[8] * B[17] * B[24] * B[33] - B[1] * B[8] * B[17] * B[27] * B[30] -
            B[1] * B[9] * B[12] * B[26] * B[35] + B[1] * B[9] * B[12] * B[29] * B[32] + B[1] * B[9] * B[14] * B[24] * B[35] - B[1] * B[9] * B[14] * B[29] * B[30] -
            B[1] * B[9] * B[17] * B[24] * B[32] + B[1] * B[9] * B[17] * B[26] * B[30] + B[1] * B[11] * B[12] * B[26] * B[33] - B[1] * B[11] * B[12] * B[27] * B[32] -
            B[1] * B[11] * B[14] * B[24] * B[33] + B[1] * B[11] * B[14] * B[27] * B[30] + B[1] * B[11] * B[15] * B[24] * B[32] - B[1] * B[11] * B[15] * B[26] * B[30] +
            B[2] * B[6] * B[13] * B[27] * B[35] - B[2] * B[6] * B[13] * B[29] * B[33] - B[2] * B[6] * B[15] * B[25] * B[35] + B[2] * B[6] * B[15] * B[29] * B[31] +
            B[2] * B[6] * B[17] * B[25] * B[33] - B[2] * B[6] * B[17] * B[27] * B[31] - B[2] * B[7] * B[12] * B[27] * B[35] + B[2] * B[7] * B[12] * B[29] * B[33] +
            B[2] * B[7] * B[15] * B[24] * B[35] - B[2] * B[7] * B[15] * B[29] * B[30] - B[2] * B[7] * B[17] * B[24] * B[33] + B[2] * B[7] * B[17] * B[27] * B[30] +
            B[2] * B[9] * B[12] * B[25] * B[35] - B[2] * B[9] * B[12] * B[29] * B[31] - B[2] * B[9] * B[13] * B[24] * B[35] + B[2] * B[9] * B[13] * B[29] * B[30] +
            B[2] * B[9] * B[17] * B[24] * B[31] - B[2] * B[9] * B[17] * B[25] * B[30] - B[2] * B[11] * B[12] * B[25] * B[33] + B[2] * B[11] * B[12] * B[27] * B[31] +
            B[2] * B[11] * B[13] * B[24] * B[33] - B[2] * B[11] * B[13] * B[27] * B[30] - B[2] * B[11] * B[15] * B[24] * B[31] + B[2] * B[11] * B[15] * B[25] * B[30] -
            B[3] * B[6] * B[13] * B[26] * B[35] + B[3] * B[6] * B[13] * B[29] * B[32] + B[3] * B[6] * B[14] * B[25] * B[35] - B[3] * B[6] * B[14] * B[29] * B[31] -
            B[3] * B[6] * B[17] * B[25] * B[32] + B[3] * B[6] * B[17] * B[26] * B[31] + B[3] * B[7] * B[12] * B[26] * B[35] - B[3] * B[7] * B[12] * B[29] * B[32] -
            B[3] * B[7] * B[14] * B[24] * B[35] + B[3] * B[7] * B[14] * B[29] * B[30] + B[3] * B[7] * B[17] * B[24] * B[32] - B[3] * B[7] * B[17] * B[26] * B[30] -
            B[3] * B[8] * B[12] * B[25] * B[35] + B[3] * B[8] * B[12] * B[29] * B[31] + B[3] * B[8] * B[13] * B[24] * B[35] - B[3] * B[8] * B[13] * B[29] * B[30] -
            B[3] * B[8] * B[17] * B[24] * B[31] + B[3] * B[8] * B[17] * B[25] * B[30] + B[3] * B[11] * B[12] * B[25] * B[32] - B[3] * B[11] * B[12] * B[26] * B[31] -
            B[3] * B[11] * B[13] * B[24] * B[32] + B[3] * B[11] * B[13] * B[26] * B[30] + B[3] * B[11] * B[14] * B[24] * B[31] - B[3] * B[11] * B[14] * B[25] * B[30] +
            B[5] * B[6] * B[13] * B[26] * B[33] - B[5] * B[6] * B[13] * B[27] * B[32] - B[5] * B[6] * B[14] * B[25] * B[33] + B[5] * B[6] * B[14] * B[27] * B[31] +
            B[5] * B[6] * B[15] * B[25] * B[32] - B[5] * B[6] * B[15] * B[26] * B[31] - B[5] * B[7] * B[12] * B[26] * B[33] + B[5] * B[7] * B[12] * B[27] * B[32] +
            B[5] * B[7] * B[14] * B[24] * B[33] - B[5] * B[7] * B[14] * B[27] * B[30] - B[5] * B[7] * B[15] * B[24] * B[32] + B[5] * B[7] * B[15] * B[26] * B[30] +
            B[5] * B[8] * B[12] * B[25] * B[33] - B[5] * B[8] * B[12] * B[27] * B[31] - B[5] * B[8] * B[13] * B[24] * B[33] + B[5] * B[8] * B[13] * B[27] * B[30] +
            B[5] * B[8] * B[15] * B[24] * B[31] - B[5] * B[8] * B[15] * B[25] * B[30] - B[5] * B[9] * B[12] * B[25] * B[32] + B[5] * B[9] * B[12] * B[26] * B[31] +
            B[5] * B[9] * B[13] * B[24] * B[32] - B[5] * B[9] * B[13] * B[26] * B[30] - B[5] * B[9] * B[14] * B[24] * B[31] + B[5] * B[9] * B[14] * B[25] * B[30] -
            B[6] * B[13] * B[20] * B[27] * B[35] + B[6] * B[13] * B[20] * B[29] * B[33] + B[6] * B[13] * B[21] * B[26] * B[35] - B[6] * B[13] * B[21] * B[29] * B[32] -
            B[6] * B[13] * B[23] * B[26] * B[33] + B[6] * B[13] * B[23] * B[27] * B[32] + B[6] * B[14] * B[19] * B[27] * B[35] - B[6] * B[14] * B[19] * B[29] * B[33] -
            B[6] * B[14] * B[21] * B[25] * B[35] + B[6] * B[14] * B[21] * B[29] * B[31] + B[6] * B[14] * B[23] * B[25] * B[33] - B[6] * B[14] * B[23] * B[27] * B[31] -
            B[6] * B[15] * B[19] * B[26] * B[35] + B[6] * B[15] * B[19] * B[29] * B[32] + B[6] * B[15] * B[20] * B[25] * B[35] - B[6] * B[15] * B[20] * B[29] * B[31] -
            B[6] * B[15] * B[23] * B[25] * B[32] + B[6] * B[15] * B[23] * B[26] * B[31] + B[6] * B[17] * B[19] * B[26] * B[33] - B[6] * B[17] * B[19] * B[27] * B[32] -
            B[6] * B[17] * B[20] * B[25] * B[33] + B[6] * B[17] * B[20] * B[27] * B[31] + B[6] * B[17] * B[21] * B[25] * B[32] - B[6] * B[17] * B[21] * B[26] * B[31] +
            B[7] * B[12] * B[20] * B[27] * B[35] - B[7] * B[12] * B[20] * B[29] * B[33] - B[7] * B[12] * B[21] * B[26] * B[35] + B[7] * B[12] * B[21] * B[29] * B[32] +
            B[7] * B[12] * B[23] * B[26] * B[33] - B[7] * B[12] * B[23] * B[27] * B[32] - B[7] * B[14] * B[18] * B[27] * B[35] + B[7] * B[14] * B[18] * B[29] * B[33] +
            B[7] * B[14] * B[21] * B[24] * B[35] - B[7] * B[14] * B[21] * B[29] * B[30] - B[7] * B[14] * B[23] * B[24] * B[33] + B[7] * B[14] * B[23] * B[27] * B[30] +
            B[7] * B[15] * B[18] * B[26] * B[35] - B[7] * B[15] * B[18] * B[29] * B[32] - B[7] * B[15] * B[20] * B[24] * B[35] + B[7] * B[15] * B[20] * B[29] * B[30] +
            B[7] * B[15] * B[23] * B[24] * B[32] - B[7] * B[15] * B[23] * B[26] * B[30] - B[7] * B[17] * B[18] * B[26] * B[33] + B[7] * B[17] * B[18] * B[27] * B[32] +
            B[7] * B[17] * B[20] * B[24] * B[33] - B[7] * B[17] * B[20] * B[27] * B[30] - B[7] * B[17] * B[21] * B[24] * B[32] + B[7] * B[17] * B[21] * B[26] * B[30] -
            B[8] * B[12] * B[19] * B[27] * B[35] + B[8] * B[12] * B[19] * B[29] * B[33] + B[8] * B[12] * B[21] * B[25] * B[35] - B[8] * B[12] * B[21] * B[29] * B[31] -
            B[8] * B[12] * B[23] * B[25] * B[33] + B[8] * B[12] * B[23] * B[27] * B[31] + B[8] * B[13] * B[18] * B[27] * B[35] - B[8] * B[13] * B[18] * B[29] * B[33] -
            B[8] * B[13] * B[21] * B[24] * B[35] + B[8] * B[13] * B[21] * B[29] * B[30] + B[8] * B[13] * B[23] * B[24] * B[33] - B[8] * B[13] * B[23] * B[27] * B[30] -
            B[8] * B[15] * B[18] * B[25] * B[35] + B[8] * B[15] * B[18] * B[29] * B[31] + B[8] * B[15] * B[19] * B[24] * B[35] - B[8] * B[15] * B[19] * B[29] * B[30] -
            B[8] * B[15] * B[23] * B[24] * B[31] + B[8] * B[15] * B[23] * B[25] * B[30] + B[8] * B[17] * B[18] * B[25] * B[33] - B[8] * B[17] * B[18] * B[27] * B[31] -
            B[8] * B[17] * B[19] * B[24] * B[33] + B[8] * B[17] * B[19] * B[27] * B[30] + B[8] * B[17] * B[21] * B[24] * B[31] - B[8] * B[17] * B[21] * B[25] * B[30] +
            B[9] * B[12] * B[19] * B[26] * B[35] - B[9] * B[12] * B[19] * B[29] * B[32] - B[9] * B[12] * B[20] * B[25] * B[35] + B[9] * B[12] * B[20] * B[29] * B[31] +
            B[9] * B[12] * B[23] * B[25] * B[32] - B[9] * B[12] * B[23] * B[26] * B[31] - B[9] * B[13] * B[18] * B[26] * B[35] + B[9] * B[13] * B[18] * B[29] * B[32] +
            B[9] * B[13] * B[20] * B[24] * B[35] - B[9] * B[13] * B[20] * B[29] * B[30] - B[9] * B[13] * B[23] * B[24] * B[32] + B[9] * B[13] * B[23] * B[26] * B[30] +
            B[9] * B[14] * B[18] * B[25] * B[35] - B[9] * B[14] * B[18] * B[29] * B[31] - B[9] * B[14] * B[19] * B[24] * B[35] + B[9] * B[14] * B[19] * B[29] * B[30] +
            B[9] * B[14] * B[23] * B[24] * B[31] - B[9] * B[14] * B[23] * B[25] * B[30] - B[9] * B[17] * B[18] * B[25] * B[32] + B[9] * B[17] * B[18] * B[26] * B[31] +
            B[9] * B[17] * B[19] * B[24] * B[32] - B[9] * B[17] * B[19] * B[26] * B[30] - B[9] * B[17] * B[20] * B[24] * B[31] + B[9] * B[17] * B[20] * B[25] * B[30] -
            B[11] * B[12] * B[19] * B[26] * B[33] + B[11] * B[12] * B[19] * B[27] * B[32] + B[11] * B[12] * B[20] * B[25] * B[33] - B[11] * B[12] * B[20] * B[27] * B[31] -
            B[11] * B[12] * B[21] * B[25] * B[32] + B[11] * B[12] * B[21] * B[26] * B[31] + B[11] * B[13] * B[18] * B[26] * B[33] - B[11] * B[13] * B[18] * B[27] * B[32] -
            B[11] * B[13] * B[20] * B[24] * B[33] + B[11] * B[13] * B[20] * B[27] * B[30] + B[11] * B[13] * B[21] * B[24] * B[32] - B[11] * B[13] * B[21] * B[26] * B[30] -
            B[11] * B[14] * B[18] * B[25] * B[33] + B[11] * B[14] * B[18] * B[27] * B[31] + B[11] * B[14] * B[19] * B[24] * B[33] - B[11] * B[14] * B[19] * B[27] * B[30] -
            B[11] * B[14] * B[21] * B[24] * B[31] + B[11] * B[14] * B[21] * B[25] * B[30] + B[11] * B[15] * B[18] * B[25] * B[32] - B[11] * B[15] * B[18] * B[26] * B[31] -
            B[11] * B[15] * B[19] * B[24] * B[32] + B[11] * B[15] * B[19] * B[26] * B[30] + B[11] * B[15] * B[20] * B[24] * B[31] - B[11] * B[15] * B[20] * B[25] * B[30]) /
           (DET);

    L[5] = -(B[0] * B[7] * B[14] * B[21] * B[28] - B[0] * B[7] * B[14] * B[22] * B[27] - B[0] * B[7] * B[15] * B[20] * B[28] + B[0] * B[7] * B[15] * B[22] * B[26] +
             B[0] * B[7] * B[16] * B[20] * B[27] - B[0] * B[7] * B[16] * B[21] * B[26] - B[0] * B[8] * B[13] * B[21] * B[28] + B[0] * B[8] * B[13] * B[22] * B[27] +
             B[0] * B[8] * B[15] * B[19] * B[28] - B[0] * B[8] * B[15] * B[22] * B[25] - B[0] * B[8] * B[16] * B[19] * B[27] + B[0] * B[8] * B[16] * B[21] * B[25] +
             B[0] * B[9] * B[13] * B[20] * B[28] - B[0] * B[9] * B[13] * B[22] * B[26] - B[0] * B[9] * B[14] * B[19] * B[28] + B[0] * B[9] * B[14] * B[22] * B[25] +
             B[0] * B[9] * B[16] * B[19] * B[26] - B[0] * B[9] * B[16] * B[20] * B[25] - B[0] * B[10] * B[13] * B[20] * B[27] + B[0] * B[10] * B[13] * B[21] * B[26] +
             B[0] * B[10] * B[14] * B[19] * B[27] - B[0] * B[10] * B[14] * B[21] * B[25] - B[0] * B[10] * B[15] * B[19] * B[26] + B[0] * B[10] * B[15] * B[20] * B[25] -
             B[1] * B[6] * B[14] * B[21] * B[28] + B[1] * B[6] * B[14] * B[22] * B[27] + B[1] * B[6] * B[15] * B[20] * B[28] - B[1] * B[6] * B[15] * B[22] * B[26] -
             B[1] * B[6] * B[16] * B[20] * B[27] + B[1] * B[6] * B[16] * B[21] * B[26] + B[1] * B[8] * B[12] * B[21] * B[28] - B[1] * B[8] * B[12] * B[22] * B[27] -
             B[1] * B[8] * B[15] * B[18] * B[28] + B[1] * B[8] * B[15] * B[22] * B[24] + B[1] * B[8] * B[16] * B[18] * B[27] - B[1] * B[8] * B[16] * B[21] * B[24] -
             B[1] * B[9] * B[12] * B[20] * B[28] + B[1] * B[9] * B[12] * B[22] * B[26] + B[1] * B[9] * B[14] * B[18] * B[28] - B[1] * B[9] * B[14] * B[22] * B[24] -
             B[1] * B[9] * B[16] * B[18] * B[26] + B[1] * B[9] * B[16] * B[20] * B[24] + B[1] * B[10] * B[12] * B[20] * B[27] - B[1] * B[10] * B[12] * B[21] * B[26] -
             B[1] * B[10] * B[14] * B[18] * B[27] + B[1] * B[10] * B[14] * B[21] * B[24] + B[1] * B[10] * B[15] * B[18] * B[26] - B[1] * B[10] * B[15] * B[20] * B[24] +
             B[2] * B[6] * B[13] * B[21] * B[28] - B[2] * B[6] * B[13] * B[22] * B[27] - B[2] * B[6] * B[15] * B[19] * B[28] + B[2] * B[6] * B[15] * B[22] * B[25] +
             B[2] * B[6] * B[16] * B[19] * B[27] - B[2] * B[6] * B[16] * B[21] * B[25] - B[2] * B[7] * B[12] * B[21] * B[28] + B[2] * B[7] * B[12] * B[22] * B[27] +
             B[2] * B[7] * B[15] * B[18] * B[28] - B[2] * B[7] * B[15] * B[22] * B[24] - B[2] * B[7] * B[16] * B[18] * B[27] + B[2] * B[7] * B[16] * B[21] * B[24] +
             B[2] * B[9] * B[12] * B[19] * B[28] - B[2] * B[9] * B[12] * B[22] * B[25] - B[2] * B[9] * B[13] * B[18] * B[28] + B[2] * B[9] * B[13] * B[22] * B[24] +
             B[2] * B[9] * B[16] * B[18] * B[25] - B[2] * B[9] * B[16] * B[19] * B[24] - B[2] * B[10] * B[12] * B[19] * B[27] + B[2] * B[10] * B[12] * B[21] * B[25] +
             B[2] * B[10] * B[13] * B[18] * B[27] - B[2] * B[10] * B[13] * B[21] * B[24] - B[2] * B[10] * B[15] * B[18] * B[25] + B[2] * B[10] * B[15] * B[19] * B[24] -
             B[3] * B[6] * B[13] * B[20] * B[28] + B[3] * B[6] * B[13] * B[22] * B[26] + B[3] * B[6] * B[14] * B[19] * B[28] - B[3] * B[6] * B[14] * B[22] * B[25] -
             B[3] * B[6] * B[16] * B[19] * B[26] + B[3] * B[6] * B[16] * B[20] * B[25] + B[3] * B[7] * B[12] * B[20] * B[28] - B[3] * B[7] * B[12] * B[22] * B[26] -
             B[3] * B[7] * B[14] * B[18] * B[28] + B[3] * B[7] * B[14] * B[22] * B[24] + B[3] * B[7] * B[16] * B[18] * B[26] - B[3] * B[7] * B[16] * B[20] * B[24] -
             B[3] * B[8] * B[12] * B[19] * B[28] + B[3] * B[8] * B[12] * B[22] * B[25] + B[3] * B[8] * B[13] * B[18] * B[28] - B[3] * B[8] * B[13] * B[22] * B[24] -
             B[3] * B[8] * B[16] * B[18] * B[25] + B[3] * B[8] * B[16] * B[19] * B[24] + B[3] * B[10] * B[12] * B[19] * B[26] - B[3] * B[10] * B[12] * B[20] * B[25] -
             B[3] * B[10] * B[13] * B[18] * B[26] + B[3] * B[10] * B[13] * B[20] * B[24] + B[3] * B[10] * B[14] * B[18] * B[25] - B[3] * B[10] * B[14] * B[19] * B[24] +
             B[4] * B[6] * B[13] * B[20] * B[27] - B[4] * B[6] * B[13] * B[21] * B[26] - B[4] * B[6] * B[14] * B[19] * B[27] + B[4] * B[6] * B[14] * B[21] * B[25] +
             B[4] * B[6] * B[15] * B[19] * B[26] - B[4] * B[6] * B[15] * B[20] * B[25] - B[4] * B[7] * B[12] * B[20] * B[27] + B[4] * B[7] * B[12] * B[21] * B[26] +
             B[4] * B[7] * B[14] * B[18] * B[27] - B[4] * B[7] * B[14] * B[21] * B[24] - B[4] * B[7] * B[15] * B[18] * B[26] + B[4] * B[7] * B[15] * B[20] * B[24] +
             B[4] * B[8] * B[12] * B[19] * B[27] - B[4] * B[8] * B[12] * B[21] * B[25] - B[4] * B[8] * B[13] * B[18] * B[27] + B[4] * B[8] * B[13] * B[21] * B[24] +
             B[4] * B[8] * B[15] * B[18] * B[25] - B[4] * B[8] * B[15] * B[19] * B[24] - B[4] * B[9] * B[12] * B[19] * B[26] + B[4] * B[9] * B[12] * B[20] * B[25] +
             B[4] * B[9] * B[13] * B[18] * B[26] - B[4] * B[9] * B[13] * B[20] * B[24] - B[4] * B[9] * B[14] * B[18] * B[25] + B[4] * B[9] * B[14] * B[19] * B[24] +
             B[0] * B[7] * B[14] * B[27] * B[34] - B[0] * B[7] * B[14] * B[28] * B[33] - B[0] * B[7] * B[15] * B[26] * B[34] + B[0] * B[7] * B[15] * B[28] * B[32] +
             B[0] * B[7] * B[16] * B[26] * B[33] - B[0] * B[7] * B[16] * B[27] * B[32] - B[0] * B[8] * B[13] * B[27] * B[34] + B[0] * B[8] * B[13] * B[28] * B[33] +
             B[0] * B[8] * B[15] * B[25] * B[34] - B[0] * B[8] * B[15] * B[28] * B[31] - B[0] * B[8] * B[16] * B[25] * B[33] + B[0] * B[8] * B[16] * B[27] * B[31] +
             B[0] * B[9] * B[13] * B[26] * B[34] - B[0] * B[9] * B[13] * B[28] * B[32] - B[0] * B[9] * B[14] * B[25] * B[34] + B[0] * B[9] * B[14] * B[28] * B[31] +
             B[0] * B[9] * B[16] * B[25] * B[32] - B[0] * B[9] * B[16] * B[26] * B[31] - B[0] * B[10] * B[13] * B[26] * B[33] + B[0] * B[10] * B[13] * B[27] * B[32] +
             B[0] * B[10] * B[14] * B[25] * B[33] - B[0] * B[10] * B[14] * B[27] * B[31] - B[0] * B[10] * B[15] * B[25] * B[32] + B[0] * B[10] * B[15] * B[26] * B[31] -
             B[1] * B[6] * B[14] * B[27] * B[34] + B[1] * B[6] * B[14] * B[28] * B[33] + B[1] * B[6] * B[15] * B[26] * B[34] - B[1] * B[6] * B[15] * B[28] * B[32] -
             B[1] * B[6] * B[16] * B[26] * B[33] + B[1] * B[6] * B[16] * B[27] * B[32] + B[1] * B[8] * B[12] * B[27] * B[34] - B[1] * B[8] * B[12] * B[28] * B[33] -
             B[1] * B[8] * B[15] * B[24] * B[34] + B[1] * B[8] * B[15] * B[28] * B[30] + B[1] * B[8] * B[16] * B[24] * B[33] - B[1] * B[8] * B[16] * B[27] * B[30] -
             B[1] * B[9] * B[12] * B[26] * B[34] + B[1] * B[9] * B[12] * B[28] * B[32] + B[1] * B[9] * B[14] * B[24] * B[34] - B[1] * B[9] * B[14] * B[28] * B[30] -
             B[1] * B[9] * B[16] * B[24] * B[32] + B[1] * B[9] * B[16] * B[26] * B[30] + B[1] * B[10] * B[12] * B[26] * B[33] - B[1] * B[10] * B[12] * B[27] * B[32] -
             B[1] * B[10] * B[14] * B[24] * B[33] + B[1] * B[10] * B[14] * B[27] * B[30] + B[1] * B[10] * B[15] * B[24] * B[32] - B[1] * B[10] * B[15] * B[26] * B[30] +
             B[2] * B[6] * B[13] * B[27] * B[34] - B[2] * B[6] * B[13] * B[28] * B[33] - B[2] * B[6] * B[15] * B[25] * B[34] + B[2] * B[6] * B[15] * B[28] * B[31] +
             B[2] * B[6] * B[16] * B[25] * B[33] - B[2] * B[6] * B[16] * B[27] * B[31] - B[2] * B[7] * B[12] * B[27] * B[34] + B[2] * B[7] * B[12] * B[28] * B[33] +
             B[2] * B[7] * B[15] * B[24] * B[34] - B[2] * B[7] * B[15] * B[28] * B[30] - B[2] * B[7] * B[16] * B[24] * B[33] + B[2] * B[7] * B[16] * B[27] * B[30] +
             B[2] * B[9] * B[12] * B[25] * B[34] - B[2] * B[9] * B[12] * B[28] * B[31] - B[2] * B[9] * B[13] * B[24] * B[34] + B[2] * B[9] * B[13] * B[28] * B[30] +
             B[2] * B[9] * B[16] * B[24] * B[31] - B[2] * B[9] * B[16] * B[25] * B[30] - B[2] * B[10] * B[12] * B[25] * B[33] + B[2] * B[10] * B[12] * B[27] * B[31] +
             B[2] * B[10] * B[13] * B[24] * B[33] - B[2] * B[10] * B[13] * B[27] * B[30] - B[2] * B[10] * B[15] * B[24] * B[31] + B[2] * B[10] * B[15] * B[25] * B[30] -
             B[3] * B[6] * B[13] * B[26] * B[34] + B[3] * B[6] * B[13] * B[28] * B[32] + B[3] * B[6] * B[14] * B[25] * B[34] - B[3] * B[6] * B[14] * B[28] * B[31] -
             B[3] * B[6] * B[16] * B[25] * B[32] + B[3] * B[6] * B[16] * B[26] * B[31] + B[3] * B[7] * B[12] * B[26] * B[34] - B[3] * B[7] * B[12] * B[28] * B[32] -
             B[3] * B[7] * B[14] * B[24] * B[34] + B[3] * B[7] * B[14] * B[28] * B[30] + B[3] * B[7] * B[16] * B[24] * B[32] - B[3] * B[7] * B[16] * B[26] * B[30] -
             B[3] * B[8] * B[12] * B[25] * B[34] + B[3] * B[8] * B[12] * B[28] * B[31] + B[3] * B[8] * B[13] * B[24] * B[34] - B[3] * B[8] * B[13] * B[28] * B[30] -
             B[3] * B[8] * B[16] * B[24] * B[31] + B[3] * B[8] * B[16] * B[25] * B[30] + B[3] * B[10] * B[12] * B[25] * B[32] - B[3] * B[10] * B[12] * B[26] * B[31] -
             B[3] * B[10] * B[13] * B[24] * B[32] + B[3] * B[10] * B[13] * B[26] * B[30] + B[3] * B[10] * B[14] * B[24] * B[31] - B[3] * B[10] * B[14] * B[25] * B[30] +
             B[4] * B[6] * B[13] * B[26] * B[33] - B[4] * B[6] * B[13] * B[27] * B[32] - B[4] * B[6] * B[14] * B[25] * B[33] + B[4] * B[6] * B[14] * B[27] * B[31] +
             B[4] * B[6] * B[15] * B[25] * B[32] - B[4] * B[6] * B[15] * B[26] * B[31] - B[4] * B[7] * B[12] * B[26] * B[33] + B[4] * B[7] * B[12] * B[27] * B[32] +
             B[4] * B[7] * B[14] * B[24] * B[33] - B[4] * B[7] * B[14] * B[27] * B[30] - B[4] * B[7] * B[15] * B[24] * B[32] + B[4] * B[7] * B[15] * B[26] * B[30] +
             B[4] * B[8] * B[12] * B[25] * B[33] - B[4] * B[8] * B[12] * B[27] * B[31] - B[4] * B[8] * B[13] * B[24] * B[33] + B[4] * B[8] * B[13] * B[27] * B[30] +
             B[4] * B[8] * B[15] * B[24] * B[31] - B[4] * B[8] * B[15] * B[25] * B[30] - B[4] * B[9] * B[12] * B[25] * B[32] + B[4] * B[9] * B[12] * B[26] * B[31] +
             B[4] * B[9] * B[13] * B[24] * B[32] - B[4] * B[9] * B[13] * B[26] * B[30] - B[4] * B[9] * B[14] * B[24] * B[31] + B[4] * B[9] * B[14] * B[25] * B[30] -
             B[6] * B[13] * B[20] * B[27] * B[34] + B[6] * B[13] * B[20] * B[28] * B[33] + B[6] * B[13] * B[21] * B[26] * B[34] - B[6] * B[13] * B[21] * B[28] * B[32] -
             B[6] * B[13] * B[22] * B[26] * B[33] + B[6] * B[13] * B[22] * B[27] * B[32] + B[6] * B[14] * B[19] * B[27] * B[34] - B[6] * B[14] * B[19] * B[28] * B[33] -
             B[6] * B[14] * B[21] * B[25] * B[34] + B[6] * B[14] * B[21] * B[28] * B[31] + B[6] * B[14] * B[22] * B[25] * B[33] - B[6] * B[14] * B[22] * B[27] * B[31] -
             B[6] * B[15] * B[19] * B[26] * B[34] + B[6] * B[15] * B[19] * B[28] * B[32] + B[6] * B[15] * B[20] * B[25] * B[34] - B[6] * B[15] * B[20] * B[28] * B[31] -
             B[6] * B[15] * B[22] * B[25] * B[32] + B[6] * B[15] * B[22] * B[26] * B[31] + B[6] * B[16] * B[19] * B[26] * B[33] - B[6] * B[16] * B[19] * B[27] * B[32] -
             B[6] * B[16] * B[20] * B[25] * B[33] + B[6] * B[16] * B[20] * B[27] * B[31] + B[6] * B[16] * B[21] * B[25] * B[32] - B[6] * B[16] * B[21] * B[26] * B[31] +
             B[7] * B[12] * B[20] * B[27] * B[34] - B[7] * B[12] * B[20] * B[28] * B[33] - B[7] * B[12] * B[21] * B[26] * B[34] + B[7] * B[12] * B[21] * B[28] * B[32] +
             B[7] * B[12] * B[22] * B[26] * B[33] - B[7] * B[12] * B[22] * B[27] * B[32] - B[7] * B[14] * B[18] * B[27] * B[34] + B[7] * B[14] * B[18] * B[28] * B[33] +
             B[7] * B[14] * B[21] * B[24] * B[34] - B[7] * B[14] * B[21] * B[28] * B[30] - B[7] * B[14] * B[22] * B[24] * B[33] + B[7] * B[14] * B[22] * B[27] * B[30] +
             B[7] * B[15] * B[18] * B[26] * B[34] - B[7] * B[15] * B[18] * B[28] * B[32] - B[7] * B[15] * B[20] * B[24] * B[34] + B[7] * B[15] * B[20] * B[28] * B[30] +
             B[7] * B[15] * B[22] * B[24] * B[32] - B[7] * B[15] * B[22] * B[26] * B[30] - B[7] * B[16] * B[18] * B[26] * B[33] + B[7] * B[16] * B[18] * B[27] * B[32] +
             B[7] * B[16] * B[20] * B[24] * B[33] - B[7] * B[16] * B[20] * B[27] * B[30] - B[7] * B[16] * B[21] * B[24] * B[32] + B[7] * B[16] * B[21] * B[26] * B[30] -
             B[8] * B[12] * B[19] * B[27] * B[34] + B[8] * B[12] * B[19] * B[28] * B[33] + B[8] * B[12] * B[21] * B[25] * B[34] - B[8] * B[12] * B[21] * B[28] * B[31] -
             B[8] * B[12] * B[22] * B[25] * B[33] + B[8] * B[12] * B[22] * B[27] * B[31] + B[8] * B[13] * B[18] * B[27] * B[34] - B[8] * B[13] * B[18] * B[28] * B[33] -
             B[8] * B[13] * B[21] * B[24] * B[34] + B[8] * B[13] * B[21] * B[28] * B[30] + B[8] * B[13] * B[22] * B[24] * B[33] - B[8] * B[13] * B[22] * B[27] * B[30] -
             B[8] * B[15] * B[18] * B[25] * B[34] + B[8] * B[15] * B[18] * B[28] * B[31] + B[8] * B[15] * B[19] * B[24] * B[34] - B[8] * B[15] * B[19] * B[28] * B[30] -
             B[8] * B[15] * B[22] * B[24] * B[31] + B[8] * B[15] * B[22] * B[25] * B[30] + B[8] * B[16] * B[18] * B[25] * B[33] - B[8] * B[16] * B[18] * B[27] * B[31] -
             B[8] * B[16] * B[19] * B[24] * B[33] + B[8] * B[16] * B[19] * B[27] * B[30] + B[8] * B[16] * B[21] * B[24] * B[31] - B[8] * B[16] * B[21] * B[25] * B[30] +
             B[9] * B[12] * B[19] * B[26] * B[34] - B[9] * B[12] * B[19] * B[28] * B[32] - B[9] * B[12] * B[20] * B[25] * B[34] + B[9] * B[12] * B[20] * B[28] * B[31] +
             B[9] * B[12] * B[22] * B[25] * B[32] - B[9] * B[12] * B[22] * B[26] * B[31] - B[9] * B[13] * B[18] * B[26] * B[34] + B[9] * B[13] * B[18] * B[28] * B[32] +
             B[9] * B[13] * B[20] * B[24] * B[34] - B[9] * B[13] * B[20] * B[28] * B[30] - B[9] * B[13] * B[22] * B[24] * B[32] + B[9] * B[13] * B[22] * B[26] * B[30] +
             B[9] * B[14] * B[18] * B[25] * B[34] - B[9] * B[14] * B[18] * B[28] * B[31] - B[9] * B[14] * B[19] * B[24] * B[34] + B[9] * B[14] * B[19] * B[28] * B[30] +
             B[9] * B[14] * B[22] * B[24] * B[31] - B[9] * B[14] * B[22] * B[25] * B[30] - B[9] * B[16] * B[18] * B[25] * B[32] + B[9] * B[16] * B[18] * B[26] * B[31] +
             B[9] * B[16] * B[19] * B[24] * B[32] - B[9] * B[16] * B[19] * B[26] * B[30] - B[9] * B[16] * B[20] * B[24] * B[31] + B[9] * B[16] * B[20] * B[25] * B[30] -
             B[10] * B[12] * B[19] * B[26] * B[33] + B[10] * B[12] * B[19] * B[27] * B[32] + B[10] * B[12] * B[20] * B[25] * B[33] - B[10] * B[12] * B[20] * B[27] * B[31] -
             B[10] * B[12] * B[21] * B[25] * B[32] + B[10] * B[12] * B[21] * B[26] * B[31] + B[10] * B[13] * B[18] * B[26] * B[33] - B[10] * B[13] * B[18] * B[27] * B[32] -
             B[10] * B[13] * B[20] * B[24] * B[33] + B[10] * B[13] * B[20] * B[27] * B[30] + B[10] * B[13] * B[21] * B[24] * B[32] - B[10] * B[13] * B[21] * B[26] * B[30] -
             B[10] * B[14] * B[18] * B[25] * B[33] + B[10] * B[14] * B[18] * B[27] * B[31] + B[10] * B[14] * B[19] * B[24] * B[33] - B[10] * B[14] * B[19] * B[27] * B[30] -
             B[10] * B[14] * B[21] * B[24] * B[31] + B[10] * B[14] * B[21] * B[25] * B[30] + B[10] * B[15] * B[18] * B[25] * B[32] - B[10] * B[15] * B[18] * B[26] * B[31] -
             B[10] * B[15] * B[19] * B[24] * B[32] + B[10] * B[15] * B[19] * B[26] * B[30] + B[10] * B[15] * B[20] * B[24] * B[31] - B[10] * B[15] * B[20] * B[25] * B[30]) /
           (DET);

    Real norm = 0;
    for (int i = 0; i < 6; i++)
        norm += abs(L[i]);

    if (norm > 100) {
        //        printf("Li, ");
        L[0] = 1.0;
        L[1] = 0.0;
        L[2] = 0.0;
        L[3] = 1.0;
        L[4] = 0.0;
        L[5] = 1.0;
    }
}
__device__ __inline__ void calc_G_Matrix(const Real4* __restrict__ sortedPosRad,
                                         const Real4* __restrict__ sortedRhoPreMu,
                                         Real* G_i,
                                         const uint* __restrict__ numNeighborsPerPart,
                                         const uint* __restrict__ neighborList,
                                         const uint numActive) {
    uint id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= numActive)
        return;

    uint index = id;

    if (sortedRhoPreMu[index].w > -0.5f && sortedRhoPreMu[index].w < 0.5f)
        return;

    Real3 posRadA = mR3(sortedPosRad[index]);
    Real SuppRadii = paramsD.h_multiplier * paramsD.h;
    Real SqRadii = SuppRadii * SuppRadii;

    // get address in grid
    int3 gridPos = calcGridPos(posRadA);
    // This is the elements of inverse of G
    Real mGi[9] = {0};

    uint NLStart = numNeighborsPerPart[index];
    uint NLEnd = numNeighborsPerPart[index + 1];
    // examine neighboring cells
    for (int n = NLStart; n < NLEnd; n++) {
        uint j = neighborList[n];
        if (j == index) {
            continue;
        }
        Real3 posRadB = mR3(sortedPosRad[j]);
        Real3 rij = Distance(posRadA, posRadB);
        Real dd = rij.x * rij.x + rij.y * rij.y + rij.z * rij.z;
        if (dd > SqRadii || sortedRhoPreMu[j].w < -1.5)
            continue;
        Real3 grad_i_wij = GradW3h(paramsD.kernel_type, rij, paramsD.ooh);
        Real3 grw_vj = grad_i_wij * paramsD.volume0;
        mGi[0] -= rij.x * grw_vj.x;
        mGi[1] -= rij.x * grw_vj.y;
        mGi[2] -= rij.x * grw_vj.z;
        mGi[3] -= rij.y * grw_vj.x;
        mGi[4] -= rij.y * grw_vj.y;
        mGi[5] -= rij.y * grw_vj.z;
        mGi[6] -= rij.z * grw_vj.x;
        mGi[7] -= rij.z * grw_vj.y;
        mGi[8] -= rij.z * grw_vj.z;
    }

    Real Det = (mGi[0] * mGi[4] * mGi[8] - mGi[0] * mGi[5] * mGi[7] - mGi[1] * mGi[3] * mGi[8] + mGi[1] * mGi[5] * mGi[6] + mGi[2] * mGi[3] * mGi[7] - mGi[2] * mGi[4] * mGi[6]);
    if (abs(Det) > 0.01) {
        Real OneOverDet = 1 / Det;
        G_i[0] = (mGi[4] * mGi[8] - mGi[5] * mGi[7]) * OneOverDet;
        G_i[1] = -(mGi[1] * mGi[8] - mGi[2] * mGi[7]) * OneOverDet;
        G_i[2] = (mGi[1] * mGi[5] - mGi[2] * mGi[4]) * OneOverDet;
        G_i[3] = -(mGi[3] * mGi[8] - mGi[5] * mGi[6]) * OneOverDet;
        G_i[4] = (mGi[0] * mGi[8] - mGi[2] * mGi[6]) * OneOverDet;
        G_i[5] = -(mGi[0] * mGi[5] - mGi[2] * mGi[3]) * OneOverDet;
        G_i[6] = (mGi[3] * mGi[7] - mGi[4] * mGi[6]) * OneOverDet;
        G_i[7] = -(mGi[0] * mGi[7] - mGi[1] * mGi[6]) * OneOverDet;
        G_i[8] = (mGi[0] * mGi[4] - mGi[1] * mGi[3]) * OneOverDet;
    } else {
        for (int i = 0; i < 9; i++) {
            G_i[i] = 0;
        }
        G_i[0] = 1;
        G_i[4] = 1;
        G_i[8] = 1;
    }
}
__device__ __inline__ void calc_A_Matrix(const Real4* __restrict__ sortedPosRad,
                                         const Real4* __restrict__ sortedRhoPreMu,
                                         Real* A_i,
                                         Real* G_i,
                                         const uint* __restrict__ numNeighborsPerPart,
                                         const uint* __restrict__ neighborList,
                                         const uint numActive) {
    uint id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= numActive)
        return;

    uint index = id;

    if (sortedRhoPreMu[index].w > -0.5f && sortedRhoPreMu[index].w < 0.5f)
        return;

    Real3 posRadA = mR3(sortedPosRad[index]);
    Real SuppRadii = paramsD.h_multiplier * paramsD.h;
    Real SqRadii = SuppRadii * SuppRadii;

    // get address in grid
    int3 gridPos = calcGridPos(posRadA);

    uint NLStart = numNeighborsPerPart[index];
    uint NLEnd = numNeighborsPerPart[index + 1];
    // examine neighboring cells
    for (int n = NLStart; n < NLEnd; n++) {
        uint j = neighborList[n];
        if (j == index) {
            continue;
        }
        Real3 posRadB = mR3(sortedPosRad[j]);
        Real3 rij = Distance(posRadA, posRadB);
        Real dd = rij.x * rij.x + rij.y * rij.y + rij.z * rij.z;
        if (dd > SqRadii || sortedRhoPreMu[j].w < -1.5)
            continue;
        Real3 grad_ij = GradW3h(paramsD.kernel_type, rij, paramsD.ooh);
        Real V_j = paramsD.markerMass / paramsD.rho0;
        Real com_part = 0;
        com_part = (G_i[0] * grad_ij.x + G_i[1] * grad_ij.y + G_i[2] * grad_ij.z) * V_j;
        A_i[0] += rij.x * rij.x * com_part;  // 111
        A_i[1] += rij.x * rij.y * com_part;  // 112
        A_i[2] += rij.x * rij.z * com_part;  // 113
        A_i[3] += rij.y * rij.x * com_part;  // 121
        A_i[4] += rij.y * rij.y * com_part;  // 122
        A_i[5] += rij.y * rij.z * com_part;  // 123
        A_i[6] += rij.z * rij.x * com_part;  // 131
        A_i[7] += rij.z * rij.y * com_part;  // 132
        A_i[8] += rij.z * rij.z * com_part;  // 133
        com_part = (G_i[3] * grad_ij.x + G_i[4] * grad_ij.y + G_i[5] * grad_ij.z) * V_j;
        A_i[9] += rij.x * rij.x * com_part;   // 211
        A_i[10] += rij.x * rij.y * com_part;  // 212
        A_i[11] += rij.x * rij.z * com_part;  // 213
        A_i[12] += rij.y * rij.x * com_part;  // 221
        A_i[13] += rij.y * rij.y * com_part;  // 222
        A_i[14] += rij.y * rij.z * com_part;  // 223
        A_i[15] += rij.z * rij.x * com_part;  // 231
        A_i[16] += rij.z * rij.y * com_part;  // 232
        A_i[17] += rij.z * rij.z * com_part;  // 233
        com_part = (G_i[6] * grad_ij.x + G_i[7] * grad_ij.y + G_i[8] * grad_ij.z) * V_j;
        A_i[18] += rij.x * rij.x * com_part;  // 311
        A_i[19] += rij.x * rij.y * com_part;  // 312
        A_i[20] += rij.x * rij.z * com_part;  // 313
        A_i[21] += rij.y * rij.x * com_part;  // 321
        A_i[22] += rij.y * rij.y * com_part;  // 322
        A_i[23] += rij.y * rij.z * com_part;  // 323
        A_i[24] += rij.z * rij.x * com_part;  // 331
        A_i[25] += rij.z * rij.y * com_part;  // 332
        A_i[26] += rij.z * rij.z * com_part;  // 333
    }
}
__device__ __inline__ void calc_L_Matrix(const Real4* __restrict__ sortedPosRad,
                                         const Real4* __restrict__ sortedRhoPreMu,
                                         Real* A_i,
                                         Real* L_i,
                                         Real* G_i,
                                         const uint* __restrict__ numNeighborsPerPart,
                                         const uint* __restrict__ neighborList,
                                         const uint numActive) {
    uint id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= numActive)
        return;

    uint index = id;

    if (sortedRhoPreMu[index].w > -0.5f && sortedRhoPreMu[index].w < 0.5f)
        return;

    Real3 posRadA = mR3(sortedPosRad[index]);
    Real SuppRadii = paramsD.h_multiplier * paramsD.h;
    Real SqRadii = SuppRadii * SuppRadii;

    Real B[36] = {0};
    Real L[6] = {0};

    // get address in grid
    int3 gridPos = calcGridPos(posRadA);
    uint NLStart = numNeighborsPerPart[index];
    uint NLEnd = numNeighborsPerPart[index + 1];
    // examine neighboring cells
    for (int n = NLStart; n < NLEnd; n++) {
        uint j = neighborList[n];
        if (j == index) {
            continue;
        }
        Real3 posRadB = mR3(sortedPosRad[j]);
        Real3 rij = Distance(posRadA, posRadB);
        Real dd = rij.x * rij.x + rij.y * rij.y + rij.z * rij.z;
        if (dd > SqRadii || sortedRhoPreMu[j].w < -1.5)
            continue;
        Real d = length(rij);
        Real3 eij = rij / d;

        Real3 grad_ij = GradW3h(paramsD.kernel_type, rij, paramsD.ooh);
        Real V_j = paramsD.markerMass / paramsD.rho0;
        Real com_part = 0;
        // mn=11

        Real XX = (eij.x * grad_ij.x);
        Real XY = (eij.x * grad_ij.y + eij.y * grad_ij.x);
        Real XZ = (eij.x * grad_ij.z + eij.z * grad_ij.x);
        Real YY = (eij.y * grad_ij.y);
        Real YZ = (eij.y * grad_ij.z + eij.z * grad_ij.y);
        Real ZZ = (eij.z * grad_ij.z);

        com_part = (A_i[0] * eij.x + A_i[9] * eij.y + A_i[18] * eij.z + rij.x * eij.x) * V_j;
        B[6 * 0 + 0] += com_part * XX;  // 11
        B[6 * 0 + 1] += com_part * XY;  // 12
        B[6 * 0 + 2] += com_part * XZ;  // 13
        B[6 * 0 + 3] += com_part * YY;  // 14
        B[6 * 0 + 4] += com_part * YZ;  // 15
        B[6 * 0 + 5] += com_part * ZZ;  // 15
        // mn=12
        com_part = (A_i[1] * eij.x + A_i[10] * eij.y + A_i[19] * eij.z + rij.x * eij.y) * V_j;
        B[6 * 1 + 0] += com_part * XX;  // 21
        B[6 * 1 + 1] += com_part * XY;  // 22
        B[6 * 1 + 2] += com_part * XZ;  // 23
        B[6 * 1 + 3] += com_part * YY;  // 24
        B[6 * 1 + 4] += com_part * YZ;  // 25
        B[6 * 1 + 5] += com_part * ZZ;  // 25

        // mn=13
        com_part = (A_i[2] * eij.x + A_i[11] * eij.y + A_i[20] * eij.z + rij.x * eij.z) * V_j;
        B[6 * 2 + 0] += com_part * XX;  // 31
        B[6 * 2 + 1] += com_part * XY;  // 32
        B[6 * 2 + 2] += com_part * XZ;  // 33
        B[6 * 2 + 3] += com_part * YY;  // 34
        B[6 * 2 + 4] += com_part * YZ;  // 35
        B[6 * 2 + 5] += com_part * ZZ;  // 36

        // Note that we skip mn=21 since it is similar to mn=12
        // mn=22
        com_part = (A_i[4] * eij.x + A_i[13] * eij.y + A_i[22] * eij.z + rij.y * eij.y) * V_j;
        B[6 * 3 + 0] += com_part * XX;  // 41
        B[6 * 3 + 1] += com_part * XY;  // 42
        B[6 * 3 + 2] += com_part * XZ;  // 43
        B[6 * 3 + 3] += com_part * YY;  // 44
        B[6 * 3 + 4] += com_part * YZ;  // 45
        B[6 * 3 + 5] += com_part * ZZ;  // 46

        // mn=23
        com_part = (A_i[5] * eij.x + A_i[14] * eij.y + A_i[23] * eij.z + rij.y * eij.z) * V_j;
        B[6 * 4 + 0] += com_part * XX;  // 51
        B[6 * 4 + 1] += com_part * XY;  // 52
        B[6 * 4 + 2] += com_part * XZ;  // 53
        B[6 * 4 + 3] += com_part * YY;  // 54
        B[6 * 4 + 4] += com_part * YZ;  // 55
        B[6 * 4 + 5] += com_part * ZZ;  // 56
        // mn=33
        com_part = (A_i[8] * eij.x + A_i[17] * eij.y + A_i[26] * eij.z + rij.z * eij.z) * V_j;
        B[6 * 5 + 0] += com_part * XX;  // 61
        B[6 * 5 + 1] += com_part * XY;  // 62
        B[6 * 5 + 2] += com_part * XZ;  // 63
        B[6 * 5 + 3] += com_part * YY;  // 64
        B[6 * 5 + 4] += com_part * YZ;  // 65
        B[6 * 5 + 5] += com_part * ZZ;  // 66
    }

    inv6xdelta_mn(B, L);
    L_i[0] = L[0];
    L_i[1] = L[1];
    L_i[2] = L[2];
    L_i[3] = L[1];
    L_i[4] = L[3];
    L_i[5] = L[4];
    L_i[6] = L[2];
    L_i[7] = L[4];
    L_i[8] = L[5];

    // Real Det = (L_i[0] * L_i[4] * L_i[8] - L_i[0] * L_i[5] * L_i[7] - L_i[1] * L_i[3] * L_i[8] +
    //             L_i[1] * L_i[5] * L_i[6] + L_i[2] * L_i[3] * L_i[7] - L_i[2] * L_i[4] * L_i[6]);
    // if (abs(Det) < 0.01) {
    //     for (int i = 0; i < 9; i++) {
    //         L_i[0 * 9 + i] = 0;
    //         L_i[0 * 9 + 0] = 1;
    //         L_i[0 * 9 + 4] = 1;
    //         L_i[0 * 9 + 8] = 1;
    //     }
    // }
    // printf("L Det %f\n", Det);
}
__device__ inline Real3 GradientOperator(float G_i[9], Real3 dist3, Real4 posRadA, Real4 posRadB, Real fA, Real fB, Real4 rhoPresMuA, Real4 rhoPresMuB) {
    Real3 gradW = GradW3h(paramsD.kernel_type, dist3, paramsD.ooh);
    Real3 gradW_new;
    gradW_new.x = G_i[0] * gradW.x + G_i[1] * gradW.y + G_i[2] * gradW.z;
    gradW_new.y = G_i[3] * gradW.x + G_i[4] * gradW.y + G_i[5] * gradW.z;
    gradW_new.z = G_i[6] * gradW.x + G_i[7] * gradW.y + G_i[8] * gradW.z;

    Real Vol = paramsD.markerMass / rhoPresMuB.x;
    Real fji = fB - fA;
    Real Gra_ij_x = fji * gradW_new.x * Vol;
    Real Gra_ij_y = fji * gradW_new.y * Vol;
    Real Gra_ij_z = fji * gradW_new.z * Vol;

    return mR3(Gra_ij_x, Gra_ij_y, Gra_ij_z);
}
__device__ inline Real4 LaplacianOperator(float G_i[9], float L_i[9], Real3 dist3, Real4 posRadA, Real4 posRadB, Real fA, Real fB, Real4 rhoPresMuA, Real4 rhoPresMuB) {
    Real3 gradW = GradW3h(paramsD.kernel_type, dist3, paramsD.ooh);
    Real d = length(dist3);
    Real3 eij = dist3 / d;

    Real Vol = paramsD.markerMass / rhoPresMuB.x;
    Real fij = fA - fB;

    Real ex_Gwx = eij.x * gradW.x;
    Real ex_Gwy = eij.x * gradW.y;
    Real ex_Gwz = eij.x * gradW.z;
    Real ey_Gwx = eij.y * gradW.x;
    Real ey_Gwy = eij.y * gradW.y;
    Real ey_Gwz = eij.y * gradW.z;
    Real ez_Gwx = eij.z * gradW.x;
    Real ez_Gwy = eij.z * gradW.y;
    Real ez_Gwz = eij.z * gradW.z;

    Real Part1 = L_i[0] * ex_Gwx + L_i[1] * ex_Gwy + L_i[2] * ex_Gwz + L_i[3] * ey_Gwx + L_i[4] * ey_Gwy + L_i[5] * ey_Gwz + L_i[6] * ez_Gwx + L_i[7] * ez_Gwy + L_i[8] * ez_Gwz;
    Real Part2 = fij / d * Vol;
    Real3 Part3 = mR3(-eij.x, -eij.y, -eij.z) * Vol;

    return mR4(2 * Part1 * Part2, Part3.x * (2 * Part1), Part3.y * (2 * Part1), Part3.z * (2 * Part1));
}
__device__ inline Real4
CfdCalcDvDt_D(Real3 gradW, Real3 dist3, Real d, Real4 posRadA, Real4 posRadB, Real3 velMasA, Real3 velMasB, Real4 rhoPresMuA, Real4 rhoPresMuB, Real* max_vel_diff) {
    if (IsBceMarker(rhoPresMuA.w) && IsBceMarker(rhoPresMuB.w))
        return mR4(0);

    // Continuity equation
    Real derivRho = paramsD.markerMass * dot(velMasA - velMasB, gradW);

    if (paramsD.use_delta_sph) {
        // diffusion term in continuity equation, this helps smoothing out the large oscillation in pressure
        // field see S. Marrone et al., "delta-SPH model for simulating violent impact flows", Computer Methods in
        // Applied Mechanics and Engineering, 200(2011), pp 1526 --1542.
        Real Psi = paramsD.density_delta * paramsD.h * paramsD.Cs * paramsD.markerMass / rhoPresMuB.x * 2. * (rhoPresMuA.x - rhoPresMuB.x) /
                   (d * d + paramsD.epsMinMarkersDis * paramsD.h * paramsD.h);
        derivRho += Psi * dot(dist3, gradW);
    }

    Real3 derivV;
    Real vAB_dot_rAB = dot(velMasA - velMasB, dist3);
    Real intermediate = vAB_dot_rAB / (d * d + paramsD.epsMinMarkersDis * paramsD.h * paramsD.h);
    if (IsFluidParticle(rhoPresMuB.w)) {
        *max_vel_diff = fmax(*max_vel_diff, fabs(paramsD.h * intermediate));
    }
    switch (paramsD.viscosity_method) {
        case ViscosityMethod::ARTIFICIAL_UNILATERAL: {
            //  pressure component
            derivV = -paramsD.markerMass * (rhoPresMuA.y / (rhoPresMuA.x * rhoPresMuA.x) + rhoPresMuB.y / (rhoPresMuB.x * rhoPresMuB.x)) * gradW;

            // artificial viscosity part, see Monaghan 1997, mainly for water
            if (vAB_dot_rAB < 0) {
                Real mu_ab = paramsD.h * vAB_dot_rAB / (d * d + paramsD.epsMinMarkersDis * paramsD.h * paramsD.h);
                Real Pi_ab = -paramsD.artificial_viscosity * paramsD.Cs * 2. / (rhoPresMuA.x + rhoPresMuB.x) * paramsD.markerMass * mu_ab;
                derivV.x -= Pi_ab * gradW.x;
                derivV.y -= Pi_ab * gradW.y;
                derivV.z -= Pi_ab * gradW.z;
            }
            break;
        }
        case ViscosityMethod::LAMINAR: {
            // laminar physics-based viscosity, directly from the Momentum equation, see Arman's PhD thesis, eq.(2.12)
            // and Morris et al.,"Modeling Low Reynolds Number Incompressible Flows Using SPH, 1997" suitable for
            // Poiseulle flow, or oil, honey, etc
            Real rAB_Dot_GradWh = dot(dist3, gradW);
            Real rAB_Dot_GradWh_OverDist = rAB_Dot_GradWh / (d * d + paramsD.epsMinMarkersDis * paramsD.h * paramsD.h);
            derivV = -paramsD.markerMass * (rhoPresMuA.y / (rhoPresMuA.x * rhoPresMuA.x) + rhoPresMuB.y / (rhoPresMuB.x * rhoPresMuB.x)) * gradW +
                     paramsD.markerMass * 8.0f * paramsD.mu0 * rAB_Dot_GradWh_OverDist * (velMasA - velMasB) / square(rhoPresMuA.x + rhoPresMuB.x);
            break;
        }
    }
    return mR4(derivV, derivRho);
}
__global__ void CfdCalcRHS_D(Real4* __restrict__ sortedDerivVelRho,
                             const Real4* __restrict__ sortedPosRad,
                             const Real3* __restrict__ sortedVelMas,
                             const Real4* __restrict__ sortedRhoPreMu,
                             const uint* __restrict__ numNeighborsPerPart,
                             const uint* __restrict__ neighborList,
                             const uint numActive,
                             uint* __restrict__ sortedFreeSurfaceIdD,
                             Real* __restrict__ sortedPosDivergence,
                             Real* __restrict__ courantViscousTimeStep,
                             Real* __restrict__ accelerationTimeStep,
                             volatile bool* error_flag) {
    uint id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= numActive)
        return;

    uint index = id;

    if (IsBceWallMarker(sortedRhoPreMu[index].w)) {
        sortedDerivVelRho[index] = mR4(0);
        return;
    }

    Real3 posRadA = mR3(sortedPosRad[index]);
    Real3 velMasA = sortedVelMas[index];
    Real4 rhoPresMuA = sortedRhoPreMu[index];
    Real4 derivVelRho = mR4(0);
    Real SuppRadii = paramsD.h_multiplier * paramsD.h;
    Real SqRadii = SuppRadii * SuppRadii;

    uint NLStart = numNeighborsPerPart[index];
    uint NLEnd = numNeighborsPerPart[index + 1];

    const bool is_sph_particle = IsSphParticle(rhoPresMuA.w);
    const bool is_fluid_particle = IsFluidParticle(rhoPresMuA.w);

    Real G_i[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    Real L_i[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    if (paramsD.use_consistent_gradient_discretization)
        calc_G_Matrix(sortedPosRad, sortedRhoPreMu, G_i, numNeighborsPerPart, neighborList, numActive);

    if (paramsD.use_consistent_laplacian_discretization) {
        Real A_i[27] = {0};
        calc_A_Matrix(sortedPosRad, sortedRhoPreMu, A_i, G_i, numNeighborsPerPart, neighborList, numActive);
        calc_L_Matrix(sortedPosRad, sortedRhoPreMu, A_i, L_i, G_i, numNeighborsPerPart, neighborList, numActive);
    }
    float Gi[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    float Li[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    Gi[0] = G_i[0];
    Gi[1] = G_i[1];
    Gi[2] = G_i[2];
    Gi[3] = G_i[3];
    Gi[4] = G_i[4];
    Gi[5] = G_i[5];
    Gi[6] = G_i[6];
    Gi[7] = G_i[7];
    Gi[8] = G_i[8];
    Li[0] = L_i[0];
    Li[1] = L_i[1];
    Li[2] = L_i[2];
    Li[3] = L_i[3];
    Li[4] = L_i[4];
    Li[5] = L_i[5];
    Li[6] = L_i[6];
    Li[7] = L_i[7];
    Li[8] = L_i[8];

    Real3 preGra = mR3(0);
    Real3 velxGra = mR3(0);
    Real3 velyGra = mR3(0);
    Real3 velzGra = mR3(0);
    Real4 velxLap = mR4(0);
    Real4 velyLap = mR4(0);
    Real4 velzLap = mR4(0);

    // get address in grid
    int3 gridPos = calcGridPos(posRadA);
    Real sum_w_i = W3h(paramsD.kernel_type, 0, paramsD.ooh) * paramsD.volume0;
    Real max_vel_diff = 0;

    Real nabla_r = 0;

    for (int n = NLStart; n < NLEnd; n++) {
        uint j = neighborList[n];
        if (j == index) {
            continue;
        }
        Real3 posRadB = mR3(sortedPosRad[j]);
        Real3 dist3 = Distance(posRadA, posRadB);
        Real dd = dist3.x * dist3.x + dist3.y * dist3.y + dist3.z * dist3.z;
        if (dd > SqRadii)
            continue;

        Real4 rhoPresMuB = sortedRhoPreMu[j];
        // no solid-solid force
        if (IsBceMarker(rhoPresMuA.w) && IsBceMarker(rhoPresMuB.w))
            continue;

        Real d = length(dist3);

        Real3 gradW = GradW3h(paramsD.kernel_type, dist3, paramsD.ooh);
        if (d > paramsD.h * Real(1.0e-9))
            nabla_r += paramsD.volume0 * dot(-dist3, gradW);

        ////modifyPressure(rhoPresMuB, dist3Alpha);
        ////if (!IsFinite(rhoPresMuB)) {
        ////    printf("ERROR (CfdCalcRHS_D): particle rhoPresMuB is NaN.\n");
        ////    *error_flag = true;
        ////}
        Real3 velMasB = sortedVelMas[j];

        derivVelRho += CfdCalcDvDt_D(gradW, dist3, d, sortedPosRad[index], sortedPosRad[j], velMasA, velMasB, rhoPresMuA, rhoPresMuB, &max_vel_diff);

        if (paramsD.use_consistent_gradient_discretization && paramsD.use_consistent_laplacian_discretization) {
            preGra += GradientOperator(Gi, dist3, sortedPosRad[index], sortedPosRad[j], -rhoPresMuA.y, rhoPresMuB.y, rhoPresMuA, rhoPresMuB);
            velxGra += GradientOperator(Gi, dist3, sortedPosRad[index], sortedPosRad[j], velMasA.x, velMasB.x, rhoPresMuA, rhoPresMuB);
            velyGra += GradientOperator(Gi, dist3, sortedPosRad[index], sortedPosRad[j], velMasA.y, velMasB.y, rhoPresMuA, rhoPresMuB);
            velzGra += GradientOperator(Gi, dist3, sortedPosRad[index], sortedPosRad[j], velMasA.z, velMasB.z, rhoPresMuA, rhoPresMuB);
            velxLap += LaplacianOperator(Gi, Li, dist3, sortedPosRad[index], sortedPosRad[j], velMasA.x, velMasB.x, rhoPresMuA, rhoPresMuB);
            velyLap += LaplacianOperator(Gi, Li, dist3, sortedPosRad[index], sortedPosRad[j], velMasA.y, velMasB.y, rhoPresMuA, rhoPresMuB);
            velzLap += LaplacianOperator(Gi, Li, dist3, sortedPosRad[index], sortedPosRad[j], velMasA.z, velMasB.z, rhoPresMuA, rhoPresMuB);
            if (d > paramsD.h * 1.0e-9)
                sum_w_i += W3h(paramsD.kernel_type, d, paramsD.ooh) * paramsD.volume0;
        }
    }

    if (paramsD.use_consistent_gradient_discretization && paramsD.use_consistent_laplacian_discretization) {
        Real nu = paramsD.mu0 / paramsD.rho0;
        Real dvxdt = -preGra.x / rhoPresMuA.x + (velxLap.x + velxGra.x * velxLap.y + velxGra.y * velxLap.z + velxGra.z * velxLap.w) * nu;
        Real dvydt = -preGra.y / rhoPresMuA.x + (velyLap.x + velyGra.x * velyLap.y + velyGra.y * velyLap.z + velyGra.z * velyLap.w) * nu;
        Real dvzdt = -preGra.z / rhoPresMuA.x + (velzLap.x + velzGra.x * velzLap.y + velzGra.y * velzLap.z + velzGra.z * velzLap.w) * nu;
        Real drhodt = -paramsD.rho0 * (velxGra.x + velyGra.y + velzGra.z);

        Real Det_G = (Gi[0] * Gi[4] * Gi[8] - Gi[0] * Gi[5] * Gi[7] - Gi[1] * Gi[3] * Gi[8] + Gi[1] * Gi[5] * Gi[6] + Gi[2] * Gi[3] * Gi[7] - Gi[2] * Gi[4] * Gi[6]);
        Real Det_L = (Li[0] * Li[4] * Li[8] - Li[0] * Li[5] * Li[7] - Li[1] * Li[3] * Li[8] + Li[1] * Li[5] * Li[6] + Li[2] * Li[3] * Li[7] - Li[2] * Li[4] * Li[6]);

        if (is_sph_particle) {
            if (Det_G > 0.9 && Det_G < 1.1 && Det_L > 0.9 && Det_L < 1.1 && sum_w_i > 0.9) {
                derivVelRho = mR4(dvxdt, dvydt, dvzdt, drhodt);
            }
        }
    }

    // Identify free-surface particles using the divergence of the position field.
    // The divergence itself is also stored, so that the shifting kernel can consume it
    // instead of accumulating the same quantity a second time.
    sortedPosDivergence[index] = nabla_r;
    sortedFreeSurfaceIdD[index] = (nabla_r < paramsD.free_surface_threshold) ? 1 : 0;

    if (!IsFinite(derivVelRho)) {
        printf("ERROR (CfdCalcRHS_D): particle derivVelRho is NaN.\n");
        *error_flag = true;
    }

    // Add gravity and other body force to fluid markers
    if (is_sph_particle) {
        Real3 totalFluidBodyForce3 = paramsD.bodyForce3 + paramsD.gravity;
        derivVelRho += mR4(totalFluidBodyForce3);
    }

    if (is_fluid_particle) {
        courantViscousTimeStep[index] = paramsD.h / (paramsD.Cs + max_vel_diff);
        Real intermediate = sqrtf(derivVelRho.x * derivVelRho.x + derivVelRho.y * derivVelRho.y + derivVelRho.z * derivVelRho.z);
        Real accT = sqrtf(paramsD.h / intermediate);
        accelerationTimeStep[index] = accT;
    }

    sortedDerivVelRho[index] = derivVelRho;
}

__host__ __device__ inline void operator*=(Real3& a, Real b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
}
__host__ __device__ inline bool IsFinite(Real3 v) {
#ifdef __CUDA_ARCH__
    return isfinite(v.x) && isfinite(v.y) && isfinite(v.z);
#else
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
#endif
}
template <ShiftingMethod SHIFT>
__device__ void ShiftingAccumulateNeighborContrib(uint index,
                                                  const Real3& posA,
                                                  const Real4& rhoPreMuA,
                                                  const Real3& velMasA,
                                                  const Real4* sortedPosRad,
                                                  const Real3* sortedVelMas,
                                                  const Real4* sortedRhoPreMu,
                                                  const uint* neighborList,
                                                  uint NLStart,
                                                  uint NLEnd,
                                                  bool consider_bce,
                                                  Real3& deltaV,
                                                  Real3& inner_sum) {
    Real SuppRadii = paramsD.h_multiplier * paramsD.h;
    Real SqRadii = SuppRadii * SuppRadii;

    // Loop over neighbors
    for (uint n = NLStart + 1; n < NLEnd; n++) {
        uint j = neighborList[n];

        // Only proceed if neighbor is fluid (this check is inlined for brevity)
        if (!IsFluidParticle(sortedRhoPreMu[j].w) && !consider_bce) {
            continue;
        }

        // Distance check
        Real3 posB = mR3(sortedPosRad[j]);
        Real3 dist3 = Distance(posA, posB);
        Real dd = dot(dist3, dist3);
        if (dd > SqRadii) {
            continue;
        }
        Real d = sqrt(dd);

        // If XSPH is required
        if constexpr (SHIFT == ShiftingMethod::XSPH || SHIFT == ShiftingMethod::PPST_XSPH) {
            Real3 velMasB = sortedVelMas[j];
            Real4 rhoPreMuB = sortedRhoPreMu[j];
            Real rho_bar = 0.5f * (rhoPreMuA.x + rhoPreMuB.x);
            deltaV += (velMasB - velMasA) * W3h(paramsD.kernel_type, d, paramsD.ooh) / rho_bar;
        }

        // If PPST is required
        if constexpr (SHIFT == ShiftingMethod::PPST || SHIFT == ShiftingMethod::PPST_XSPH) {
            // Fictitious sphere for PPST
            Real dFictitious = paramsD.d0 * Real(1.241);
            if (d < 1.25f * dFictitious) {  // TODO: If we don't put this, flexible cable crashes - why do we need this?
                Real oodFictitious = 1 / dFictitious;
                Real delta_ij = (dFictitious - d) * oodFictitious;
                Real beta = (delta_ij > 0) ? paramsD.shifting_ppst_push : paramsD.shifting_ppst_pull;
                inner_sum += beta * fmax(delta_ij, static_cast<Real>(-0.1f)) * (dist3 / d);
            }
        }

        if constexpr (SHIFT == ShiftingMethod::DIFFUSION || SHIFT == ShiftingMethod::DIFFUSION_XSPH) {
            // for diffusion based shifting, inner sum is the gradient of concentration
            Real4 rhoPreMuB = sortedRhoPreMu[j];
            inner_sum += paramsD.markerMass / rhoPreMuB.x * GradW3h(paramsD.kernel_type, dist3, paramsD.ooh);
        }
    }
}

template <ShiftingMethod SHIFT>
__global__ void Calc_Shifting_D(Real3* vel_XSPH_Sorted_D,
                                Real4* sortedPosRad,
                                Real3* sortedVelMas,
                                Real4* sortedRhoPreMu,
                                const uint* numNeighborsPerPart,
                                const uint* neighborList,
                                const uint numActive,
                                const Real* __restrict__ sortedPosDivergence,
                                volatile bool* error_flag) {
    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numActive)
        return;

    // If not fluid, do nothing
    if (!IsFluidParticle(sortedRhoPreMu[index].w))
        return;

    // Gather data
    Real4 rhoPreMuA = sortedRhoPreMu[index];
    Real3 velMasA = sortedVelMas[index];
    Real3 posA = mR3(sortedPosRad[index]);

    // Range for neighbors
    uint NLStart = numNeighborsPerPart[index];
    uint NLEnd = numNeighborsPerPart[index + 1];

    // Accumulators for different methods
    Real3 deltaV = mR3(0);
    Real3 inner_sum = mR3(0);

    bool consider_bce = false;
    if constexpr (SHIFT == ShiftingMethod::DIFFUSION || SHIFT == ShiftingMethod::DIFFUSION_XSPH) {
        consider_bce = true;
    }

    // Accumulate neighbor contribution
    ShiftingAccumulateNeighborContrib<SHIFT>(index, posA, rhoPreMuA, velMasA, sortedPosRad, sortedVelMas, sortedRhoPreMu, neighborList, NLStart, NLEnd, consider_bce, deltaV,
                                             inner_sum);

    // Post-process depending on SHIFT
    Real3 result = mR3(0);

    if constexpr (SHIFT == ShiftingMethod::XSPH) {
        result = paramsD.markerMass * paramsD.shifting_xsph_eps * deltaV;  // XSPH velocity
    } else if constexpr (SHIFT == ShiftingMethod::PPST) {
        Real vA = length(velMasA);
        Real vAdT = vA * paramsD.dT;

        // scale, limit displacement
        inner_sum = vAdT * inner_sum;
        Real upper_limit = 0.05f * vAdT;
        Real cur_len = length(inner_sum);
        if (cur_len > upper_limit) {
            inner_sum *= (upper_limit / (cur_len + 1e-9f));
        }
        result = inner_sum / paramsD.dT;  // Update as a velocity
    } else if constexpr (SHIFT == ShiftingMethod::PPST_XSPH) {
        Real vA = length(velMasA);
        Real vAdT = vA * paramsD.dT;

        // combine XSPH and PPST
        Real3 xsphVel = paramsD.shifting_xsph_eps * paramsD.markerMass * deltaV;

        inner_sum = vAdT * inner_sum;
        Real upper_limit = 0.05f * vAdT;
        Real cur_len = length(inner_sum);
        if (cur_len > upper_limit) {
            inner_sum *= (upper_limit / (cur_len + 1e-9f));
        }
        result = xsphVel + inner_sum / paramsD.dT;  // Update as a velocity
    } else if constexpr (SHIFT == ShiftingMethod::DIFFUSION) {
        Real vA = length(velMasA);
        Real AFSM = paramsD.shifting_diffusion_AFSM;
        Real AFST = paramsD.shifting_diffusion_AFST;
        Real nabla_r = sortedPosDivergence[index];

        result = -paramsD.shifting_diffusion_A * paramsD.h * inner_sum * vA;

        // Taper the shift as the kernel support degrades toward the free surface: no shift at or
        // below AFST, ramping linearly up to the full shift at AFSM. Note that the previous form
        // applied the ramp only for nabla_r < AFST, where the factor is negative (reaching -AFST as
        // nabla_r -> 0 with the default AFSM - AFST = 1): the shift of the very particles the taper
        // targets was reversed in direction and grew without bound, while the AFST..AFSM band the
        // taper is meant to attenuate was left at the full shift.
        Real ramp = (nabla_r - AFST) / (AFSM - AFST);
        result = result * fmin(fmax(ramp, Real(0)), Real(1));

    } else if constexpr (SHIFT == ShiftingMethod::DIFFUSION_XSPH) {
        Real vA = length(velMasA);
        Real AFSM = paramsD.shifting_diffusion_AFSM;
        Real AFST = paramsD.shifting_diffusion_AFST;
        Real nabla_r = sortedPosDivergence[index];

        // For now, just add the contribution from XSPH and Diffusion
        Real3 xsphVel = paramsD.shifting_xsph_eps * paramsD.markerMass * deltaV;

        result = -paramsD.shifting_diffusion_A * paramsD.h * inner_sum * vA;

        // See the DIFFUSION branch above for the rationale of this taper
        Real ramp = (nabla_r - AFST) / (AFSM - AFST);
        result = result * fmin(fmax(ramp, Real(0)), Real(1));

        result = xsphVel + result;
    }

    // Write out - This is a velocity
    vel_XSPH_Sorted_D[index] = result;

    // Check for NaNs
    if (!IsFinite(result)) {
        printf("Error! Shifting produce  NAN. Particle: %u\n", index);
        *error_flag = true;
    }
}

static constexpr double CH_1_3 = 1.0 / 3.0;

__host__ __device__ inline void operator-=(Real3& a, Real3 b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
__host__ __device__ inline void operator/=(Real3& a, Real b) {
    a.x /= b;
    a.y /= b;
    a.z /= b;
}
inline __device__ Real Eos(Real rho, EosType eos_type) {
    switch (eos_type) {
        case EosType::TAIT: {
            // Tait EOS with Hughes and Graham Correction
            // See https://pysph.readthedocs.io/en/latest/reference/equations.html#basic-wcsph-equations
            // if (rho < paramsD.rho0)
            //      rho = paramsD.rho0;
            Real gama = 7;
            Real B = paramsD.rho0 * paramsD.Cs * paramsD.Cs / gama;
            return B * (pow(rho / paramsD.rho0, gama) - 1) + paramsD.base_pressure;
        }
        case EosType::ISOTHERMAL: {
            // Isothermal equation of state
            return paramsD.Cs * paramsD.Cs * (rho - paramsD.rho0);
        }
    }
    return -1;
}
__device__ void PositionEulerStep(Real dT, const Real3& vel, Real4& pos) {
    Real3 p = mR3(pos);
    p += dT * vel;
    pos = mR4(p, pos.w);
}
__device__ void VelocityEulerStep(Real dT, const Real3& acc, Real3& vel) {
    vel += dT * acc;
}
__device__ void DensityEulerStep(Real dT, const Real& deriv, EosType eos, Real4& rho_p) {
    rho_p.x += dT * deriv;
    rho_p.y = Eos(rho_p.x, eos);
}
__device__ void TauEulerStep(Real dT,
                             const Real3& deriv_tau_diag,
                             const Real3& deriv_tau_offdiag,
                             const Real& deriv_rho,
                             bool close_to_surface,
                             Real3& tau_diag,
                             Real3& tau_offdiag,
                             Real4& rho_p,
                             Real3& pcEvSv,
                             volatile bool* error_flag) {
    if (paramsD.rheology_model_crm == RheologyCRM::MU_OF_I) {
        Real3 new_tau_diag = tau_diag + dT * deriv_tau_diag;
        Real3 new_tau_offdiag = tau_offdiag + dT * deriv_tau_offdiag;

        // Check for plastic flow
        Real p_n = -CH_1_3 * (tau_diag.x + tau_diag.y + tau_diag.z);
        Real p_tr = -CH_1_3 * (new_tau_diag.x + new_tau_diag.y + new_tau_diag.z);
        // Tau now becomes the deviatoric component
        // reusing the same register so names get confusing
        tau_diag += mR3(p_n);
        new_tau_diag += mR3(p_tr);

        Real tau_n = square(tau_diag.x) + square(tau_diag.y) + square(tau_diag.z) +                             //
                     2 * (square(tau_offdiag.x) + square(tau_offdiag.y) + square(tau_offdiag.z));               //
        Real tau_tr = square(new_tau_diag.x) + square(new_tau_diag.y) + square(new_tau_diag.z) +                //
                      2 * (square(new_tau_offdiag.x) + square(new_tau_offdiag.y) + square(new_tau_offdiag.z));  //
        tau_n = sqrt(0.5 * tau_n);
        tau_tr = sqrt(0.5 * tau_tr);
        Real Chi = abs(tau_tr - tau_n) * paramsD.INV_G_shear / dT;

        // Should use the positive magnitude according to "A constitutive law for dense granular flows" Nature 2006
        Real mu_s = paramsD.mu_fric_s;
        Real mu_2 = paramsD.mu_fric_2;
        // Real s_0 = mu_s * p_tr;
        // Real s_2 = mu_2 * p_tr;
        // Real xi = 1.1;
        Real dia = paramsD.ave_diam;
        Real I0 = paramsD.mu_I0;  // xi*dia*sqrt(rhoPresMu.x);//

        // Zero-tension cutoff (Mohr-Coulomb tension cut-off convention): cohesion
        // contributes shear strength through tau_max = mu * p + c below, but grants
        // no tension capacity. The previous cutoff at p_tr < -c/mu_s let particles
        // sustain negative pressure, which (a) drives the SPH tensile instability
        // (particle clumping that destabilizes geostatic stress states and collapses
        // bearing responses whenever c > 0), and (b) made the inertial number I NaN
        // for p_tr < 0 (sqrt of a negative), silently disabling the yield check for
        // tensile particles. For c = 0 this update is identical to the previous one.
        if (p_tr < Real(0))
            p_tr = Real(0);

        Real I = Chi * dia * sqrt(paramsD.rho0 / (p_tr + 1.0e-9));

        Real coh = paramsD.Coh_coeff;
        // Real Chi_cri = 0.1;
        // if (Chi < Chi_cri){
        //     coh = paramsD.Coh_coeff * (1.0 - sin(-1.57 + 3.14 * (Chi / Chi_cri))) / 2.0;
        //     // coh = paramsD.Coh_coeff * (1.0 - I / I_cri);
        // } else {
        //     coh = 0.0;
        // }

        Real mu = mu_s + (mu_2 - mu_s) * (I + 1.0e-9) / (I0 + I + 1.0e-9);
        // Real G0 = paramsD.G_shear;
        // Real alpha = xi*G0*I0*(dT)*sqrt(p_tr);
        // Real B0 = s_2 + tau_tr + alpha;
        // Real H0 = s_2*tau_tr + s_0*alpha;
        // Real tau_n1 = (B0+sqrt(B0*B0-4*H0))/(2*H0+1e-9);
        // if(tau_tr>s_0){
        //     Real coeff = tau_n1/(tau_tr+1e-9);
        //     updatedTauXxYyZz = updatedTauXxYyZz*coeff;
        //     updatedTauXyXzYz = updatedTauXyXzYz*coeff;
        // }
        Real tau_max = p_tr * mu + coh;  // p_tr*paramsD.Q_FA;
        // should use tau_max instead of s_0 according to
        // "A constitutive law for dense granular flows" Nature 2006
        if (tau_tr > tau_max) {
            Real coeff = tau_max / (tau_tr + 1e-9);
            new_tau_diag *= coeff;
            new_tau_offdiag *= coeff;
        }

        // Set stress to zero if the particle is close to free surface
        if (close_to_surface == 1) {
            new_tau_diag = mR3(0.0);
            new_tau_offdiag = mR3(0.0);
            p_tr = 0.0;
        }
        // Going back to sigma from the deviatoric component
        tau_diag = new_tau_diag - mR3(p_tr);
        tau_offdiag = new_tau_offdiag;

        rho_p.y = p_tr;
        // rho_p.x = rho_p.x + deriv_rho * dT;
        rho_p.x = paramsD.rho0;
    } else {
        // Implementation reference
        // https://docs.itascacg.com/flac3d700/common/models/camclay/doc/modelcamclay.html#equation-modelmcceqvn
        // N is state at next time step
        // n is state at current time step
        // N_tr is trial
        Real mcc_M = paramsD.mcc_M;
        Real mcc_lambda = paramsD.mcc_lambda;
        Real mcc_kappa = paramsD.mcc_kappa;
        Real p_c = pcEvSv.x;
        // Use the previous time steps specific volume
        Real specific_volume_n = pcEvSv.z;

        // Compute local bulk/shear moduli per MCC (Itasca Eq. (15), (46))
        // Use previous-state mean pressure with floors, clamp and under-relax for stability
        // Recomputing for now instead of storing for each particle to avoid memory usage (compute once in crmRHS)
        Real p_n = -CH_1_3 * (tau_diag.x + tau_diag.y + tau_diag.z);
        // Candidate bulk modulus from MCC
        Real K_cand = specific_volume_n * (p_n) / paramsD.mcc_kappa;
        // Clamp K
        Real K_n = fmin(fmax(K_cand, Real(0.1) * paramsD.K_bulk), Real(1.0) * paramsD.K_bulk);
        // Shear
        Real G_cand = (3.0 * K_n * (1.0 - 2.0 * paramsD.Nu_poisson)) / (2.0 * (1.0 + paramsD.Nu_poisson));
        Real G_n = fmin(fmax(G_cand, Real(0.1) * paramsD.G_shear), Real(1.0) * paramsD.G_shear);
        // Trial stress using convention N = n + 1
        Real3 sig_diag_N_tr = tau_diag + dT * deriv_tau_diag;
        Real3 sig_offdiag_N_tr = tau_offdiag + dT * deriv_tau_offdiag;
        Real p_N_tr = -CH_1_3 * (sig_diag_N_tr.x + sig_diag_N_tr.y + sig_diag_N_tr.z);
        // Deviatoric component of the trial stress
        Real3 dev_diag_N_tr = sig_diag_N_tr + mR3(p_N_tr);
        Real3 dev_offdiag_N_tr = sig_offdiag_N_tr;

        // Computing trial von misses stress (q_N_tr)
        Real inner_product = square(dev_diag_N_tr.x) + square(dev_diag_N_tr.y) + square(dev_diag_N_tr.z) +
                             2 * (square(dev_offdiag_N_tr.x) + square(dev_offdiag_N_tr.y) + square(dev_offdiag_N_tr.z));
        Real J_2 = inner_product * 0.5;
        Real q_N_tr = sqrt(3.0 * J_2);

        // Computing yield function (f_N)
        Real f_N = square(q_N_tr) + square(mcc_M) * p_N_tr * (p_N_tr - p_c);
        Real3 s_diag_N = sig_diag_N_tr;
        Real3 s_offdiag_N = sig_offdiag_N_tr;
        Real p_N = p_N_tr;
        Real delta_lambda_N = 0.0;
        Real c_v = 0.0;
        // Scale-aware tolerances
        Real f_scale = square(q_N_tr) + square(mcc_M) * square(p_N_tr);
        Real f_tol = fmax(Real(1e-12), Real(1e-6) * f_scale);
        Real q_eps = fmax(Real(1e-9), Real(1e-6) * (fabs(p_N_tr) + q_N_tr));

        // No tension
        if (p_N_tr < 0) {
            tau_diag = mR3(0.0);
            tau_offdiag = mR3(0.0);
            rho_p.y = 0.0;
        } else if (p_N_tr > 0 && f_N <= f_tol) {
            // Nearly on the yield surface: treat as elastic
            tau_diag = sig_diag_N_tr;
            tau_offdiag = sig_offdiag_N_tr;
            rho_p.y = p_N_tr;
        } else if (p_N_tr > 0 && f_N > 0) {  // If we have yielded, find the new deviatoric stress
            c_v = square(mcc_M) * (2 * p_N_tr - p_c);
            Real c_q = 2 * q_N_tr;
            // Quadratic coefficients for plastic strain increment (delta lambda)
            Real a = square(mcc_M * K_n * c_v) + square(3 * G_n * c_q);
            Real b = -K_n * square(c_v) - 3 * G_n * square(c_q);
            Real c = f_N;
            // q -> 0 guard: pure volumetric correction
            if (q_N_tr < q_eps) {
                c_q = 0.0;
                a = square(mcc_M * K_n * c_v);
                b = -K_n * square(c_v);
            }
            // Solve quadratic robustly, in double precision: in single precision b^2
            // overflows to inf once |b| > 1.84e19, which is reached for q_N_tr ~ 390 kPa
            // at clamped moduli (b ~ -3*G*(2q)^2). The inf then propagates through
            // delta_lambda to the hardening update and permanently poisons p_c.
            if (a <= 0) {
                delta_lambda_N = 0.0;
            } else {
                double ad = (double)a;
                double bd = (double)b;
                double cd = (double)c;
                double disc = fmax(bd * bd - 4.0 * ad * cd, 0.0);
                double sqrt_disc = sqrt(disc);
                double inv_2a = 0.5 / ad;
                double r1 = (-bd + sqrt_disc) * inv_2a;
                double r2 = (-bd - sqrt_disc) * inv_2a;
                // pick the smallest positive root (or 0 if none)
                double dl = 0.0;
                if (r1 > 0 && r2 > 0)
                    dl = (r1 < r2) ? r1 : r2;
                else if (r1 > 0)
                    dl = r1;
                else if (r2 > 0)
                    dl = r2;
                delta_lambda_N = (Real)dl;
            }

            // Get the mapped stress
            p_N = p_N_tr - K_n * delta_lambda_N * c_v;
            Real q_N = (q_N_tr - 3 * G_n * delta_lambda_N * c_q);
            s_diag_N = q_N * dev_diag_N_tr / (q_N_tr + q_eps);
            s_offdiag_N = q_N * dev_offdiag_N_tr / (q_N_tr + q_eps);

            // No tension allowed, else get the new stress from the mapped deviatoric stress and pressure
            if (p_N < 0) {
                tau_diag = mR3(0.0);
                tau_offdiag = mR3(0.0);
                rho_p.y = 0.0;
            } else {
                tau_diag = s_diag_N - mR3(p_N);
                tau_offdiag = s_offdiag_N;
                rho_p.y = p_N;
            }
            // Update the consolidation pressure (only if we are not close to the free surface).
            // Guard against a non-finite plastic strain increment: without this, one bad
            // delta_lambda makes p_c infinite forever (the fmax floor does not catch inf)
            // and the particle never yields again.
            Real plastic_volumentric_strain = delta_lambda_N * c_v;
            if (!close_to_surface && isfinite(plastic_volumentric_strain)) {
                pcEvSv.x *= (1 + plastic_volumentric_strain * (specific_volume_n / (mcc_lambda - mcc_kappa)));
                // pcEvSv.x *= exp(plastic_volumentric_strain * (specific_volume_n / (mcc_lambda - mcc_kappa)));
                pcEvSv.x = fmax(Real(100.0), pcEvSv.x);
            }
        }

        // If we are close to free surface, set stress tensor to zero tensor
        // TODO: Should the consolidation pressure also be zero? This makes the material have zero yield
        if (close_to_surface == 1) {
            tau_diag = mR3(0.0);
            tau_offdiag = mR3(0.0);
            rho_p.y = 0.0;
        }
        // Update density
        rho_p.x = rho_p.x + deriv_rho * dT;
        // Update specific volume based on the volumetric strain rate
        // TODO: How are we guaranteed that the volumetric strain rate and the density are synchronized?
        // One aspect is that they are both numerically integrated from the divergence of the velocity field
        pcEvSv.z *= (1 - pcEvSv.y * dT);
        // Set min to prevent collapse of the specific volume
        pcEvSv.z = fmax(Real(1.0), pcEvSv.z);
    }

    // Flag a rheology failure (non-finite updated stress state) so the host can abort.
    // This complements the host-side position/density NaN scans, which do not cover stress.
    // Nothing is evaluated when error checking is disabled (error_flag is then null).
    // Check only state the active rheology model integrates: mu(I) neither reads nor
    // updates the consolidation state (pcEvSv), which can legitimately be non-finite
    // straight out of default initialization (AddSphParticle derives the specific volume
    // with log(pc / p1), which is not finite for the default zero initial pressure).
    if (error_flag) {
        bool rheology_failed = !IsFinite(tau_diag) || !IsFinite(tau_offdiag) || !IsFinite(rho_p);
        if (paramsD.rheology_model_crm == RheologyCRM::MCC)
            rheology_failed = rheology_failed || !IsFinite(pcEvSv);
        if (rheology_failed)
            *error_flag = true;
    }
}
__global__ void EulerStep_D(Real4* posRadD,
                            Real3* velMasD,
                            Real4* rhoPresMuD,
                            Real3* tauXxYyZzD,
                            Real3* tauXyXzYzD,
                            Real3* pcEvSvD,
                            const Real3* vel_XSPH_D,
                            const Real4* derivVelRhoD,
                            const Real3* derivTauXxYyZzD,
                            const Real3* derivTauXyXzYzD,
                            const uint* freeSurfaceIdD,
                            const int32_t* activityIdentifierSortedD,
                            const uint numActive,
                            Real dT,
                            volatile bool* error_flag) {
    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numActive)
        return;

    // Only update active SPH particles, not extended active particles
    if (IsBceMarker(rhoPresMuD[index].w) || activityIdentifierSortedD[index] <= 0)
        return;

    // Euler step for position
    PositionEulerStep(dT, velMasD[index] + vel_XSPH_D[index], posRadD[index]);

    // Euler step for velocity
    VelocityEulerStep(dT, mR3(derivVelRhoD[index]), velMasD[index]);

    if (paramsD.physics_problem == PhysicsProblem::CRM) {
        // Euler step for tau and pressure update
        TauEulerStep(dT, derivTauXxYyZzD[index], derivTauXyXzYzD[index], derivVelRhoD[index].w, freeSurfaceIdD[index], tauXxYyZzD[index], tauXyXzYzD[index], rhoPresMuD[index],
                     pcEvSvD[index], error_flag);
    } else {
        // Euler step for density and pressure update from EOS
        DensityEulerStep(dT, derivVelRhoD[index].w, paramsD.eos_type, rhoPresMuD[index]);
    }
}

__global__ void ApplyPeriodicBoundaryY_D(Real4* posRadD, Real4* rhoPresMuD, const uint numActive) {
    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numActive)
        return;

    Real4 rhoPresMu = rhoPresMuD[index];
    // no need to do anything if it is a BCE marker
    if (IsBceMarker(rhoPresMu.w))
        return;

    Real3 posRad = mR3(posRadD[index]);
    Real h = posRadD[index].w;

    if (posRad.y > paramsD.cMax.y) {
        posRad.y -= (paramsD.cMax.y - paramsD.cMin.y);
        posRadD[index] = mR4(posRad, h);
        rhoPresMu.y = rhoPresMu.y + paramsD.delta_pressure.y;
        rhoPresMuD[index] = rhoPresMu;
        return;
    }
    if (posRad.y < paramsD.cMin.y) {
        posRad.y += (paramsD.cMax.y - paramsD.cMin.y);
        posRadD[index] = mR4(posRad, h);
        rhoPresMu.y = rhoPresMu.y - paramsD.delta_pressure.y;
        rhoPresMuD[index] = rhoPresMu;
        return;
    }
}